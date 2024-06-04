from torch.utils.data import DataLoader
from datasets import load_dataset
from typing import List, Dict, Callable
import numpy as np
from collections import defaultdict
from functools import partial

DEFAULT_CACHE_DIR = "~/.cache/huggingface"

def ltr_scale(x):
    """
    `log(1 + |x|) * sign(x)`
    
    Learning to rank feature can vary significantly in size,
    this scaling is proven to help.
    """
    return np.log1p(np.abs(x)) * np.sign(x)

def position_recipr(x):
    """
    `1. / x`

    The UPE paper propose two formulas for preprocessing before optimizing for position,
    of which one this yielded the best results (though on synthetical data).
    """
    return 1. / x


def collate(samples: List[Dict[str, np.ndarray]], labels: Dict[str, Callable | None], pad=True):
    max_n = max([sample["n"] for sample in samples])
    batch = defaultdict(lambda: [])

    for sample in samples:
        pad_n = max_n - sample["n"]

        for name, feature in sample.items():
            if name in labels:
                # preprocess data
                preprocess = labels[name]
                if preprocess is not None:
                    feature = preprocess(feature)

                # pad pad_n documents
                if pad:
                    padded = np.pad(feature, [(0, pad_n)] + [(0, 0)]*(feature.ndim-1))
                    batch[name].append(padded)
                else:
                    batch[name].extend(feature) # TODO: document

        if pad:
            mask = np.pad(np.ones(sample["n"]), [(0, pad_n)]).astype(bool)
            batch["mask"].append(mask)
        else:
            batch["groups"].append(sample["n"])

    return { name: np.array(x) for name, x in batch.items() }

def load_dataloader(
        name: str, split: str, batch_size: int, labels: Dict[str, Callable | None], pad=True, cache_dir=DEFAULT_CACHE_DIR
    ) -> DataLoader:
    """
    Args:
    If `batch_size == -1`, the entire dataset is returned as a single batch.\\
    If `pad`, all samples in a batch are padded to the longest `n` and a `mask` variable is set.
    Otherwise, to guarantee regularness of the output matrices (without padding), all query-document pairs are bunched into a single array. To preserve listwise information, a `groups` variable is set
    that denotes the number of documents for each query.

    Returns a dataloader that preprocesses the features in `labels` with the corresponding functions.
    """

    dataset = load_dataset(
        "philipphager/baidu-ultr_uva-mlm-ctr",
        name=name, split=split, cache_dir=cache_dir, trust_remote_code=True
    )
    dataset.set_format("numpy")

    if batch_size == -1:
        batch_size = len(dataset)

    return DataLoader(
        dataset, batch_size,
        collate_fn=partial(collate, labels=labels, pad=pad),
        num_workers=4, # TODO: optimize
        # persistent_workers=True, # TODO
        pin_memory=True
    )