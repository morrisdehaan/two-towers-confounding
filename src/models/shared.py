import flax.serialization
import numpy as np
import pickle
from utils import data
from functools import partial
from typing import Dict, List, Callable

# log1p based scaling is used on learning to rank features
LTR_FEATURES = {
    "bm25": data.ltr_scale,
    "bm25_title": data.ltr_scale,
    "bm25_abstract": data.ltr_scale,
    "tf_idf": data.ltr_scale,
    "tf": data.ltr_scale,
    "idf": data.ltr_scale,
    "ql_jelinek_mercer_short": data.ltr_scale,
    "ql_jelinek_mercer_long": data.ltr_scale,
    "ql_dirichlet": data.ltr_scale,
    "document_length": data.ltr_scale,
    "title_length": data.ltr_scale,
    "abstract_length": data.ltr_scale,
}

def one_hot(inds: np.ndarray, num_classes: int) -> np.ndarray:
    """ One hot encodes an indices array. """
    return np.identity(num_classes)[inds - 1]

def load_default_features(labels: List[str], n_ranks=None) -> Dict[str, Callable]:
    """
    Loads input features with default preprocessing functions.
    Namely, log1p scaling for learning to rank features, no op for Bert embeddings
    and one-hot encoding for positions.
    """

    features = {}
    if "ltr" in labels:
        features |= LTR_FEATURES
    if "bert" in labels:
        # bert features are normalized and don't require preprocessing
        features |= { "query_document_embedding": None }
    if "position" in labels:
        if n_ranks is None:
            raise Exception("No number of ranks 'n_rank' specified!")        
        # one hot encoding
        features |= { "position": partial(one_hot, num_classes=n_ranks) }
    return features

def save_flax_params(params, file: str):
    state_dict = flax.serialization.to_state_dict(params)
    pickle.dump(state_dict, open(file, "wb"))

def load_flax_params(file: str):
    return pickle.load(open(file, "rb"))