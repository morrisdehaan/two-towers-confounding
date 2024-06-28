from flax.training.train_state import TrainState
import jax
import optax, rax
import numpy as np
from functools import partial
from typing import Tuple, Dict, Callable
from models import shared, two_towers
from utils import data

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

def fit(config: Dict) -> Tuple[two_towers.RelevanceTower, Dict]:
    # * load data *
    features = shared.load_default_features(config["optional_features"])
    labels = { "click": None } | features

    train_loader = data.load_dataloader(
        "clicks", f"train[:{config['train_data_percent_used']}%]", config["batch_size"], labels
    )
    valid_loader = data.load_dataloader(
        "clicks", f"test[:{config['valid_data_percent_used']}%]", config["batch_size"], labels
    )

    # * init model *
    init_key = jax.random.key(0)

    model = two_towers.RelevanceTower(config["hyperparams"]["hidden"], features)
    params = model.init(init_key, next(iter(valid_loader)))

    # * train model *
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(config["hyperparams"]["learning_rate"])
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

def load(params_path: str, config: Dict) -> Tuple[two_towers.RelevanceTower, Dict]:
    features = shared.load_default_features(config["optional_features"])
    model = two_towers.RelevanceTower(config["hyperparams"]["hidden"], features)
    return model, shared.load_flax_params(params_path)

# TODO: make shared
def evaluate(model: two_towers.RelevanceTower, params: Dict, top_n: int, data_percent: int):
    labels = { "label": None } | model.features
    test_loader = data.load_dataloader("annotations", f"test[:{data_percent}%]", batch_size=512, labels=labels)

    ndcgs, dcgs = [], []
    for batch in test_loader:
        pred = model.apply(params, x=batch)
        ndcgs.extend(rax.ndcg_metric(pred, batch["label"], where=batch["mask"], topn=top_n, reduce_fn=None))
        dcgs.extend(rax.dcg_metric(pred, batch["label"], where=batch["mask"], topn=top_n, reduce_fn=None))

    return { f"mean ndcg@{top_n}": np.mean(ndcgs), f"mean dcg@{top_n}": np.mean(dcgs) }
