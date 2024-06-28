import lightgbm
import rax
import numpy as np
from utils import data
from models import shared
from typing import List, Dict, Callable, Sequence, Tuple

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
    
def fit_logging_policy(config: Dict) -> BoosterRanker:
    # TODO: use all features
    # * load data into a single batch *
    features = shared.load_default_features(config["optional_features"])

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

def fold_iter(data_size: int, kfolds: int) -> Sequence[Tuple[int, int]]:
    """ Returns an iterator that yields the range [start, end) of the ith fold. """

    assert kfolds > 1, "There must be more than 1 fold!"
    
    fold_size = int(data_size / kfolds) + 1
    return zip(
        # start index
        range(0, data_size, fold_size),
        # end index
        map(lambda s: min(s, data_size-1), range(fold_size, data_size + fold_size, fold_size))
    )


def fit_expert_labels(config: Dict) -> List[BoosterRanker]:
    """ Fits on expert annotations directly using k-fold cross validation """

    # * load data into a single batch *
    features = shared.load_default_features(config["optional_features"])

    # lightGBM only accepts integers, only relative ordering matters anyway
    labels = { "label": None } | features

    train_loader = data.load_dataloader(
        "annotations", "test", batch_size=-1, labels=labels, pad=False
    )
    full_data = next(iter(train_loader))

    n_queries = len(full_data["groups"])

    rankers = []
    for query_start, query_end in fold_iter(n_queries, config["k-folds"]):
        start = np.sum(full_data["groups"][:query_start])
        end = start + np.sum(full_data["groups"][query_start:query_end])

        train_data = {
            key: np.concatenate(
                (full_data[key][:start], full_data[key][end:])
            ) for key in labels
        } | { "groups": np.concatenate((full_data["groups"][:query_start], full_data["groups"][query_end:])) }

        # * train ranker *
        # initialize
        lambdamart_ranker = lightgbm.LGBMRanker(**config["hyperparams"])

        train_x = np.concatenate([atleast_2d(train_data[name]) for name in features.keys()], axis=-1)
        train_y, train_groups = train_data["label"], train_data["groups"]

        # train ranker, NOTE: it copies train data
        lambdamart_ranker.fit(train_x, train_y, group=train_groups)
        rankers.append(BoosterRanker(lambdamart_ranker.booster_, features))

        # TODO: remove?
        del lambdamart_ranker
    return rankers

def load(params_path: str, config: Dict) -> BoosterRanker:
    features = shared.load_default_features(config["optional_features"])

    booster = lightgbm.Booster(model_file=params_path)
    return BoosterRanker(booster, features)
