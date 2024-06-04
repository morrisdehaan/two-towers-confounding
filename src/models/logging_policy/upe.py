""" Proposed model from the 'Unconfounded Propensity Estimation for Unbiased Ranking' (2023) paper. """

import flax.training
import flax.training.train_state
import jax, jax.numpy as jnp
import numpy as np
import flax, flax.linen as nn
import optax, rax
import pickle, pandas as pd
from utils import data, loss as loss_fns
from functools import partial
from typing import Dict, Sequence, Callable, Tuple, Any

# TODO: not like in paper
class ConfoundEnc(nn.Module):
    """ Confounder Encoder.  """

    encoder_output_size: int
    features: Sequence[str]

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array]):
        x = jnp.concat([jnp.atleast_3d(x[name]) for name in self.features], axis=-1)
        return nn.Dense(self.encoder_output_size)(x)

# TODO: should dropout be used, even when it's frozen? (I don't think so, what's the point?)
class FFNN(nn.Module):
    hidden: Sequence[int]
    dropout_rate: float

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array], training: bool) -> jax.Array:
        for n in self.hidden:
            x = nn.Dense(n)(x)
            x = nn.elu(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        return jnp.squeeze(nn.Dense(1)(x))
    
class ConfounderLearner(nn.Module):
    """ Caputures confounding during the first learning cycle. """

    # TODO: rename?
    encoder_output_size: int
    features: Sequence[str]
    ffnn_hidden: Sequence[int]
    dropout_rate: float

    def setup(self):
        self.encoder = ConfoundEnc(self.encoder_output_size, self.features)
        self.ffnn = FFNN(self.ffnn_hidden, self.dropout_rate)

    def __call__(self, x: Dict[str, jax.Array], training: bool):
        return self.ffnn(self.encoder(x), training)
    
class PositionEnc(nn.Module):
    """ Position encoder. """

    n_ranks: int
    encoder_output_size: int

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array]) -> jax.Array:
        return nn.Embed(self.n_ranks, self.encoder_output_size)(x["position"])
    
class UPE(nn.Module):
    """ Models unbiased propensity estimation (UPE) in the second learning cycle. """

    n_ranks: int
    encoder_output_size: int
    confound_features: Sequence[str]
    ffnn_hidden: Sequence[str]
    
    def setup(self):
        self.position_encoder = PositionEnc(self.n_ranks, self.encoder_output_size)
        # both are pretrained and frozen
        self.confounder_encoder = ConfoundEnc(self.encoder_output_size, self.confound_features)
        self.ffnn = FFNN(self.ffnn_hidden, 0.0)

    @nn.compact
    def __call__(self, x: Dict[str, jax.Array]):
        return self.ffnn(self.confounder_encoder(x) + self.position_encoder(x), False)
       
def fit_logging_policy(features: Dict[str, Callable], config: Dict) -> Tuple[ConfounderLearner, Any]:
    # * training utils *
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

    # * load data *
    labels = { "position": data.position_recipr } | features

    train_loader = data.load_dataloader(
        "clicks", f"train[:{config['train_data_percent_used']}%]", batch_size=config["batch_size"], labels=labels
    )
    valid_loader = data.load_dataloader(
        "clicks", f"test[:{config['valid_data_percent_used']}%]", batch_size=config["batch_size"], labels=labels
    )

    # * init *
    init_key, dropout_key = jax.random.split(jax.random.key(0), 2)

    # initialize model
    model = ConfounderLearner(
        encoder_output_size=config["hyperparams"]["encoder_output_size"],
        features=features,
        ffnn_hidden=config["hyperparams"]["ffnn_hidden"],
        dropout_rate=config["hyperparams"]["dropout_rate"]
    )
    params = model.init(init_key, next(iter(train_loader)), training=True)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adagrad(learning_rate=config["hyperparams"]["learning_rate"]),
        dropout_key=dropout_key
    )
    loss_fn = loss_fns.attention_rank_loss

    # * train *
    for epoch in range(1, config["epochs"]+1):
        # train
        train_loss = 0.
        for batch in train_loader:
            x, y = batch, batch["position"]
            train_state, loss = train_step(x, y, train_state, loss_fn)
            train_loss += loss
        train_loss /= len(train_loader)

        # validate
        valid_loss = 0.
        for batch in valid_loader:
            x, y = batch, batch["position"]
            valid_loss += valid_step(x, y, train_state, loss_fn)
        valid_loss /= len(valid_loader)

        print(f"epoch {epoch}: train loss = {train_loss:.3f} - validation loss = {valid_loss:.3f}")
    return model, train_state.params

def evaluate_logging_policy(ranker: ConfounderLearner, params, top_n: int, data_percent: int) -> Dict[str, float]:
    """ Evaluates for ndcg@`top_n` and dcg@`top_n`. """

    labels = { "position": data.position_recipr } | ranker.features
    test_loader = data.load_dataloader("clicks", f"test[:{data_percent}%]", batch_size=512, labels=labels, pad=True)

    ndcgs, dcgs = [], []
    for batch in test_loader:
        pred = ranker.apply(params, batch, training=False)        
        labels = batch["position"]

        ndcgs.extend(rax.ndcg_metric(pred, labels, where=batch["mask"], topn=top_n, reduce_fn=None))
        dcgs.extend(rax.dcg_metric(pred, labels, where=batch["mask"], topn=top_n, reduce_fn=None))
    return { f"mean ndcg@{top_n}": np.mean(ndcgs), f"mean dcg@{top_n}": np.mean(dcgs) }

# TODO: move somewhere more general?
def save_params(params, file: str):
    state_dict = flax.serialization.to_state_dict(params)
    pickle.dump(state_dict, open(file, "wb"))

def load_logging_policy(params_path, features: Dict[str, Callable], config: Dict) -> Tuple[ConfounderLearner, Any]:
    model = ConfounderLearner(
        encoder_output_size=config["hyperparams"]["encoder_output_size"],
        features=features,
        ffnn_hidden=config["hyperparams"]["ffnn_hidden"],
        dropout_rate=config["hyperparams"]["dropout_rate"]
    )
    params = pickle.load(open(params_path, "rb"))
    return model, params

def fit(features: Dict[str, Callable], config: Dict) -> Tuple[ConfounderLearner, Any]:
    # TODO: lol
    class TrainState(flax.training.train_state.TrainState):
        pass

    @partial(jax.jit, static_argnames="loss_fn")
    def train_step(x, y, state: TrainState, loss_fn: Callable) -> Tuple[TrainState, jax.Array]:
        def compute_loss(params, x, y):
            pred = state.apply_fn(params, x=x)
            return loss_fn(pred, y, where=x["mask"])    
        loss, grads = jax.value_and_grad(compute_loss)(state.params, x, y)
        return state.apply_gradients(grads=grads), loss

    @partial(jax.jit, static_argnames="loss_fn")
    def valid_step(x, y, state: TrainState, loss_fn: Callable) -> jax.Array:
        pred = state.apply_fn(state.params, x=x)
        return loss_fn(pred, y, where=x["mask"])

    # * load data *
    labels = { "position": None } | features

    train_loader = data.load_dataloader(
        "clicks", f"train[:{config['train_data_percent_used']}%]", batch_size=config["batch_size"], labels=labels
    )
    valid_loader = data.load_dataloader(
        "clicks", f"test[:{config['valid_data_percent_used']}%]", batch_size=config["batch_size"], labels=labels
    )

    # * init model *
    init_key = jax.random.key(0)

    model = UPE(
        n_ranks=config["hyperparams"]["n_ranks"],
        encoder_output_size=config["hyperparams"]["encoder_output_size"],
        features=features,
        ffnn_hidden=config["hyperparams"]["ffnn_hidden"]
    )
    params = model.init(init_key, next(iter(train_loader)))

    # set pretrained params
    confound_params = pickle.load(open(config["pretrained_confounder_params_path"], "rb"))
    params["params"]["confounder_encoder"] = confound_params["params"]["encoder"]
    params["params"]["ffnn"] = confound_params["params"]["ffnn"]

    # use optimizer that freezes pretrained params
    optimizer_partitions = {
        "trainable": optax.adagrad(config["hyperparams"]["learning_rate"]),
        "frozen": optax.set_to_zero()
    }
    param_partitions = flax.traverse_util.path_aware_map(
        lambda path, _: "frozen" if "confounder_encoder" in path or "ffnn" in path else "trainable",
        params
    )
    tx = optax.multi_transform(optimizer_partitions, param_partitions)

    # load P(E|k) propensities
    # TODO: credit source + rational, find out what propensities in UPE paper are used
    propensities = pd.read_csv(config["propensities_path"])["propensity"].to_numpy()
    propensities = jnp.array(np.pad(propensities, (0, config["hyperparams"]["n_ranks"] - propensities.shape[0])))

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    loss_fn = loss_fns.attention_rank_loss

    for epoch in range(1, config["epochs"]+1):
        # train
        train_loss = 0.
        for batch in train_loader:
            x, y = batch, propensities[batch["position"]]
            train_state, loss = train_step(x, y, train_state, loss_fn)
            train_loss += loss
        train_loss /= len(train_loader)

        # validate
        valid_loss = 0.
        for batch in valid_loader:
            x, y = batch, propensities[batch["position"]]
            valid_loss += valid_step(x, y, train_state, loss_fn)
        valid_loss /= len(valid_loader)

        print(f"epoch {epoch}: train loss = {train_loss:.3f} - validation loss = {valid_loss:.3f}")
    return model, train_state.params

def evaluate(ranker: UPE, params):
    # TODO
    print("TODO: make something up")
    pass

def load(params_path, features: Dict[str, Callable], config: Dict) -> Tuple[UPE, Any]:
    model = UPE(
        n_ranks=config["hyperparams"]["n_ranks"],
        encoder_output_size=config["hyperparams"]["encoder_output_size"],
        features=features,
        ffnn_hidden=config["hyperparams"]["ffnn_hidden"]
    )
    params = pickle.load(open(params_path, "rb"))
    return model, params