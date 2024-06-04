import lightgbm
import rax
import numpy as np
from utils import data
from typing import Dict, Callable

# TODO: remove
def atleast_2d(arr: np.ndarray) -> np.ndarray:
    """ Variant of `numpy.atleast_2d`, but slightly different. """

    if len(arr.shape) == 0:
        return arr.reshape(1, 1)
    elif len(arr.shape) == 1:
        return arr[:, np.newaxis]
    elif len(arr.shape) >= 2:
        return arr

class BoosterRanker:
    def __init__(self, booster: lightgbm.Booster, features: Dict[str, Callable]) -> None:
        self._ranker = booster
        self.features = features
    
    def predict(self, input: Dict[str, np.ndarray]) -> np.ndarray:
        input = np.concatenate([np.atleast_3d(input[name]) for name in self.features.keys()], axis=-1)
        # flatten to an array of feature vectors
        input_flat = input.reshape(-1, input.shape[-1])
        # predict scores, cast shape back to an array of ranking scores
        return self._ranker.predict(input_flat).reshape(input.shape[0], input.shape[1])
    
    def save(self, file: str):
        self._ranker.save_model(file)
    
def fit_logging_policy(features: Dict[str, Callable], config: Dict) -> BoosterRanker:
    # TODO: use all features
    # * load data into a single batch *
    # TODO: remove max? # TODO: dont like the positions in the data
    # lightGBM only accepts integers, only relative ordering matters anyway
    labels = { "position": lambda k: np.maximum(10 - k, 0) } | features

    train_loader = data.load_dataloader(
        "clicks", f"train[:{config['train_data_percent_used']}%]",
        batch_size=-1, labels=labels, pad=False
    )
    train_data = next(iter(train_loader))

    # * train ranker *
    # initialize # TODO: hyperparam tuning
    lambdamart_ranker = lightgbm.LGBMRanker(**config["hyperparams"])

    train_x = np.concatenate([atleast_2d(train_data[name]) for name in features.keys()], axis=-1)
    train_y, train_groups = train_data["position"], train_data["groups"]

    # train ranker, NOTE: it copies train data
    lambdamart_ranker.fit(train_x, train_y, group=train_groups)
    return BoosterRanker(lambdamart_ranker.booster_, features)

def load_logging_policy(path, features) -> BoosterRanker:
    booster = lightgbm.Booster(model_file=path)
    return BoosterRanker(booster, features)

def evaluate_logging_policy(ranker: BoosterRanker, top_n: int, data_percent: int) -> Dict[str, float]:
    """ Evaluates for ndcg@`top_n` and dcg@`top_n`. """

    labels = { "position": data.position_recipr } | ranker.features
    test_loader = data.load_dataloader("clicks", f"test[:{data_percent}%]", batch_size=512, labels=labels, pad=True)

    ndcgs, dcgs = [], []
    for batch in test_loader:
        pred = ranker.predict(batch)        
        labels = batch["position"]

        ndcgs.extend(rax.ndcg_metric(pred, labels, where=batch["mask"], topn=top_n, reduce_fn=None))
        dcgs.extend(rax.dcg_metric(pred, labels, where=batch["mask"], topn=top_n, reduce_fn=None))

    return { f"mean ndcg@{top_n}": np.mean(ndcgs), f"mean dcg@{top_n}": np.mean(dcgs) }
    