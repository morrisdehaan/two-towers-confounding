import flax.training
import flax.training.train_state
import jax, jax.numpy as jnp
import flax, flax.linen as nn
import optax, rax
import numpy as np
import pandas as pd
from functools import partial
from typing import Sequence, Tuple, Dict, Any, Callable
from models import upe, shared
from utils import data

class BiasEmbedding(nn.Module):
    embedding_file: str

    def setup(self):
        df = pd.read_csv(self.embedding_file)
        self.embedding = jnp.array(df["propensity"].to_numpy())

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array]) -> jax.Array:
        inds = jnp.argmax(x["position"], axis=-1)
        return self.embedding[inds]

class RelevanceTower(nn.Module):
    hidden: Sequence[int]
    features: Dict[str, Callable]

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array]) -> jax.Array:
        # concatenate features
        x = jnp.concat([jnp.atleast_3d(x[name]) for name in self.features.keys()], axis=-1)

        for n in self.hidden:
            x = nn.Dense(n)(x)
            x = nn.relu(x)
        return jnp.squeeze(nn.Dense(1)(x))

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
        return nn.Dropout(self.dropout_rate, deterministic=not training or not self.use_dropout)(x)

class TwoTowers(nn.Module):
    relevance: RelevanceTower
    examine: BiasTower | upe.UPE | BiasEmbedding

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array], training: bool) -> jax.Array:
        if type(self.examine) != BiasTower:
            return self.relevance(x) + self.examine(x)
        else:
            return self.relevance(x) + self.examine(x, training)
        
class TrainState(flax.training.train_state.TrainState):
    dropout_key: jax.Array

@partial(jax.jit, static_argnames="loss_fn")
def train_step(x, y, state: TrainState, loss_fn: Callable) -> Tuple[TrainState, jax.Array]:
    dropout_key = jax.random.fold_in(state.dropout_key, state.step)

    def compute_loss(params, x, y):
        pred = state.apply_fn(params, x=x, training=True, rngs={ "dropout": dropout_key })
        return loss_fn(pred, y, where=x["mask"])
    
    loss, grads = jax.value_and_grad(compute_loss)(state.params, x, y)
    return state.apply_gradients(grads=grads), loss

@partial(jax.jit, static_argnames="loss_fn")
def valid_step(x, y, state: TrainState, loss_fn: Callable) -> jax.Array:
    pred = state.apply_fn(state.params, x=x, training=False)
    return loss_fn(pred, y, where=x["mask"])

def fit(config: Dict) -> Tuple[TwoTowers, Any]:
    # * load data *
    relevance_features = shared.load_default_features(config["optional_features"])

    if config["bias_tower_type"] == "ffnn":
        bias_features = shared.load_default_features(
            config["optional_bias_features"], n_ranks=config["hyperparams"].get("n_ranks")
        )
    elif config["bias_tower_type"] == "upe" or config["bias_tower_type"] == "embedding":
        # UPE invariably takes only position as bias input
        bias_features = shared.load_default_features(
            ["position"], n_ranks=config["hyperparams"]["n_ranks"]
        )
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
            encoder_output_size=config["hyperparams"]["encoder_output_size"],
            confound_features=relevance_features,
            ffnn_hidden=config["hyperparams"]["ffnn_hidden"]
        )

        # init model
        model = TwoTowers(relevance_tower, bias_tower)
        params = model.init(init_key, next(iter(train_loader)), training=True)

        # load pretrained upe params
        upe_params = shared.load_flax_params(config["pretrained_upe_params_path"])
        params["params"]["examine"] = upe_params["params"]

        # freeze pretrained params
        optimizer_partitions = { "trainable": tx, "frozen": optax.set_to_zero() }
        param_partitions = flax.traverse_util.path_aware_map(
            lambda path, _: "frozen" if "examine" in path else "trainable",
            params
        )
        tx = optax.multi_transform(optimizer_partitions, param_partitions)
    elif config["bias_tower_type"] == "embedding":
        bias_tower = BiasEmbedding(config["embedding_file"])

        model = TwoTowers(relevance_tower, bias_tower)
        params = model.init(init_key, next(iter(train_loader)), training=True)
    elif config["bias_tower_type"] == "ffnn":
        use_dropout = config["use_bias_dropout"]
        bias_tower = BiasTower(
            hidden=config["hyperparams"]["bias_hidden"],
            features=bias_features.keys(),
            use_dropout=use_dropout,
            dropout_rate=0.0 if not use_dropout else config["hyperparams"]["dropout_rate"]
        )

        # init model
        model = TwoTowers(relevance_tower, bias_tower)
        params = model.init(init_key, next(iter(train_loader)), training=True)
    else:
        raise Exception(f"{config['bias_tower_type']} is not a valid bias tower type!")
    
    # * train model *
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_key=dropout_key
    )

    if config["loss_fn"] == "listwise":
        loss_fn = rax.softmax_loss
    elif config["loss_fn"] == "pointwise":
        loss_fn = rax.pointwise_sigmoid_loss
    elif config["loss_fn"] == "pairwise":
        # lambdarank
        loss_fn = partial(rax.pairwise_logistic_loss, lambdaweight_fn=partial(
            rax.dcg2_lambdaweight, normalize=True, topn=config["lambdamart_topn"]
        ))
    else:
        raise Exception("Invalid loss function!")
    
    for epoch in range(1, config["epochs"]+1):
        train_loss = 0.
        for batch in train_loader:
            train_state, loss = train_step(batch, batch["click"], train_state, loss_fn)
            train_loss += loss
        train_loss /= len(train_loader)

        valid_loss = 0.
        for batch in valid_loader:
            valid_loss += valid_step(batch, batch["click"], train_state, loss_fn)
        valid_loss /= len(valid_loader)

        print(f"epoch {epoch}: train loss = {train_loss:.3f} - valid loss = {valid_loss:.3f}")
    return model, train_state.params

def load(params_path: str, config: Dict):
    relevance_features = shared.load_default_features(config["optional_features"])

    relevance_tower = RelevanceTower(
        hidden=config["hyperparams"]["hidden"],
        features=relevance_features
    )

    if config["bias_tower_type"] == "upe":
        bias_tower = upe.UPE(
            encoder_output_size=config["hyperparams"]["encoder_output_size"],
            confound_features=relevance_features,
            ffnn_hidden=config["hyperparams"]["ffnn_hidden"]
        )

        # init model
        model = TwoTowers(relevance_tower, bias_tower)
    elif config["bias_tower_type"] == "embedding":
        bias_tower = BiasEmbedding(config["embedding_file"])

        model = TwoTowers(relevance_tower, bias_tower)
    elif config["bias_tower_type"] == "ffnn":
        bias_features = shared.load_default_features(
            config["optional_bias_features"], n_ranks=config["hyperparams"].get("n_ranks")
        )
            
        use_dropout = config["use_bias_dropout"]
        bias_tower = BiasTower(
            hidden=config["hyperparams"]["bias_hidden"],
            features=bias_features,
            use_dropout=use_dropout,
            dropout_rate=0.0 if not use_dropout else config["hyperparams"]["dropout_rate"]
        )

        # init model
        model = TwoTowers(relevance_tower, bias_tower)
    else:
        raise Exception(f"{config['bias_tower_type']} is not a valid bias tower type!")
    
    return model, shared.load_flax_params(params_path)

def evaluate(model: TwoTowers, params: Dict, top_n: int, data_percent: int):
    labels = { "label": None } | model.relevance.features
    test_loader = data.load_dataloader("annotations", f"test[:{data_percent}%]", batch_size=512, labels=labels)

    ndcgs, dcgs = [], []
    for batch in test_loader:
        pred = model.relevance.apply({ "params": params["params"]["relevance"] }, x=batch)
        ndcgs.extend(rax.ndcg_metric(pred, batch["label"], where=batch["mask"], topn=top_n, reduce_fn=None))
        dcgs.extend(rax.dcg_metric(pred, batch["label"], where=batch["mask"], topn=top_n, reduce_fn=None))

    return { f"mean ndcg@{top_n}": np.mean(ndcgs), f"mean dcg@{top_n}": np.mean(dcgs) }
