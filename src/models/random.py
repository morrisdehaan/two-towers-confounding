import numpy as np
from typing import Dict

class RandomRanker:
    def __init__(self):
        # just some arbitrary feature that exists for every document-query pair
        self.feature_name = "title_length"
        self.features = { self.feature_name: None }

    def predict(self, x: Dict[str, np.ndarray]) -> np.ndarray:
        return np.random.sample(x[self.feature_name].shape)