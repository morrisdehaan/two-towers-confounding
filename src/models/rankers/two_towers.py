# TODO: clean

import flax.training
import flax.training.train_state
import jax, jax.numpy as jnp
import flax, flax.linen as nn
import optax, rax
import pickle
from functools import partial
from typing import Sequence, Tuple, Dict, Any, Callable
from models.logging_policy import upe
from utils import data

class RelevanceTower(nn.Module):
    hidden: Sequence[int]
    features: Sequence[str]

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array]) -> jax.Array:
        # concatenate features
        x = jnp.concat([jnp.atleast_3d(x[name]) for name in self.features], axis=-1)

        for n in self.hidden:
            x = nn.Dense(n)(x)
            x = nn.relu(x)
        return jnp.squeeze(nn.Dense(1)(x))

# TODO: one hot encode position in preprocessing
# TODO: only apply dropout to output layer, dropout=~0.3, does need some tuning
class BiasTower(nn.Module):
    hidden: Sequence[int]
    features: Sequence[int] # bias features

    use_dropout: bool
    dropout_rate: float

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array], training: bool) -> jax.Array:
        x = jnp.concat([jnp.atleast_3d(x[name]) for name in self.features], axis=-1)

        for n in self.hidden:
            x = nn.Dense(n)(x)
            x = nn.relu(x)
        x = jnp.squeeze(nn.Dense(1)(x))
        # dropout on bias tower output, as per the dropout technique
        return nn.Dropout(self.dropout_rate, deterministic=not training or not self.use_dropout)

class TwoTowers(nn.Module):
    relevance: RelevanceTower
    examine: BiasTower | upe.UPE

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array], training: bool) -> jax.Array:
        # UPE is already fit, therefore dropout need not apply
        if type(self.examine) == upe.UPE:
            training = False
        return nn.sigmoid(self.relevance(x) + self.examine(x, training))

class TrainState(flax.training.train_state.TrainState):
    dropout_key: jax.Array

@partial(jax.jit, static_argnames="loss_fn")
def train_step(x, y, state: TrainState, loss_fn: Callable) -> Tuple[TrainState, jax.Array]:
    dropout_key = jax.random.fold_in(state.dropout_key, state.step)

    def compute_loss(params, x, y):
        pred = state.apply_fn(
            params, x=x, training=True, rngs={ "dropout": dropout_key }
        )
        return loss_fn(pred, y, where=x["mask"])
    
    loss, grads = jax.value_and_grad(compute_loss)(state.params, x, y)
    return state.apply_gradients(grads=grads), loss

@partial(jax.jit, static_argnames="loss_fn")
def valid_step(x, y, state: TrainState, loss_fn: Callable) -> jax.Array:
    pred = state.apply_fn(
            state.params, x=x, training=False
    )
    return loss_fn(pred, y, where=x["mask"])

def fit(
        relevance_features: Dict[str, Callable], bias_features: Dict[str, Callable], config: Dict
) -> Tuple[TwoTowers, Any]:
    # * load data *
    labels = { "click": None } | relevance_features | bias_features

    train_loader = data.load_dataloader(
        "clicks", f"train[:{config['train_data_percent_used']}%]", batch_size=config["batch_size"], labels=labels
    )
    valid_loader = data.load_dataloader(
        "clicks", f"test[:{config['valid_data_percent_used']}%]", batch_size=config["batch_size"], labels=labels
    )

    # * init model *
    init_key, dropout_key = jax.random.split(jax.random.key(0))

    tx = optax.adam(config["hyperparams"]["learning_rate"])

    relevance_tower = RelevanceTower(
        hidden=config["hyperparams"]["hidden"],
        features=relevance_features
    )

    if config["bias_tower_type"] == "upe":
        bias_tower = upe.UPE(
            n_ranks=config["hyperparams"]["n_ranks"],
            encoder_output_size=config["hyperparams"]["encoder_output_size"],
            features=bias_features,
            ffnn_hidden=config["hyperparams"]["ffnn_hidden"]
        )

        # init model
        model = TwoTowers(relevance_tower, bias_tower)
        params = model.init(init_key, next(iter(train_loader)), training=True)

        # load pretrained upe params
        upe_params = pickle.load(open(config["pretrained_upe_params_path"], "rb"))
        params["params"]["examine"] = upe_params["params"]

        # freeze pretrained params
        optimizer_partitions = { "trainable": tx, "frozen": optax.set_to_zero() } # TODO: follow paper
        param_partitions = flax.traverse_util.path_aware_map(
            lambda path, _: "frozen" if "examine" in path else "trainable",
            params
        )
        tx = optax.multi_transform(optimizer_partitions, param_partitions)

    elif config["bias_tower_type"] == "ffnn":
        use_dropout = config["hyperparams"]["use_dropout"]
        bias_tower = BiasTower(
            hidden=config["hyperparams"]["hidden"],
            features=bias_features,
            use_dropout=use_dropout,
            dropout_rate=0.0 if not use_dropout else config["hyperparams"]["dropout_rate"]
        )

        # init model
        model = TwoTowers(relevance_tower, bias_tower)
        params = model.init(init_key, next(iter(train_loader)), training=True)
    else:
        raise Exception(f"'{config["bias_tower_type"]}' is not a valid bias tower type!")
    
    # * train model *
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_key=dropout_key
    )
    loss_fn = rax.softmax_loss

    for epoch in range(1,config["epochs"]+1):
        # train
        train_loss = 0.
        for batch in train_loader:
            x, y = batch, batch["click"]
            train_state, loss = train_step(x, y, train_state, loss_fn)
            train_loss += loss
        train_loss /= len(train_loader)

        # validate
        valid_loss = 0.
        for batch in valid_loader:
            x, y = batch, batch["click"]
            valid_loss += valid_step(x, y, train_state, loss_fn)
        valid_loss /= len(valid_loader)

        print(f"epoch {epoch}: train loss = {train_loss:.3f} - validation loss = {valid_loss:.3f}")
    return model, train_state.params

def evaluate():
    # TODO
    print("TODO")