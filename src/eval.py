import pandas as pd
import rax, numpy as np
from models import lambdamart, upe, two_towers, naive, random
from utils.config import load_config, load_param_path
from utils import data
from collections import defaultdict
import sys, os, warnings
from functools import partial
from typing import Tuple, Dict, List, Callable, Sequence, Tuple

TOP_N = 10
METRIC = "ndcg"

def evaluate(
        rank: Callable, features: Dict[str, Callable], label: Tuple[str, Callable], top_n: int,
        data_name: str, data_split: str, batch_size: int,
        metrics: List[str], reduce_fn: Callable | None
) -> Dict:
    """ Evaluates a ranker function `rank` on the given label. """

    metric_fns = {
        "ndcg": partial(rax.ndcg_metric, topn=top_n, reduce_fn=None),
        "dcg": partial(rax.dcg_metric, topn=top_n, reduce_fn=None)
    }
    if any([metric not in metric_fns.keys() for metric in metrics]):
        raise Exception("Unsupported metric used!")

    labels = { label[0]: label[1] } | features
    test_loader = data.load_dataloader(data_name, data_split, batch_size=batch_size, labels=labels)

    scores = { metric: [] for metric in metrics }
    for batch in test_loader:
        pred = rank(batch)
        for metric in metrics:
            scores[metric].extend(metric_fns[metric](pred, batch[label[0]], where=batch["mask"]))

    if reduce_fn is None:
        return scores
    else:
        return { metric: reduce_fn(score) for metric, score in scores.items() }

def evaluate_on_logging_policy(
        rank: Callable, features: Dict[str, Callable], top_n: int,
        metrics: List[str] = ["ndcg", "dcg"], reduce_fn: Callable | None = np.mean
) -> Dict:
    """ Evaluates a ranker function `rank` using nDCG@`top_n` and DCG@`top_n` on 1 partition of the click dataset. """

    return evaluate(
        rank, features, ("position", data.position_recipr), top_n,
        # 25% is one partition of the dataset
        data_name="clicks", data_split=f"test[:25%]", batch_size=512,
        metrics=metrics, reduce_fn=reduce_fn
    )

def evaluate_on_experts(
        # TODO: rename rank_fn
        rank: Callable, features: Dict[str, Callable], top_n: int,
        metrics: List[str] = ["ndcg", "dcg"], reduce_fn: Callable | None = np.mean
) -> Dict:
    """ Evaluates a ranker function `rank` using nDCG@`top_n` and DCG@`top_n` on the expert annotations. """

    return evaluate(
        rank, features, ("label", None), top_n,
        data_name="annotations", data_split="test", batch_size=512,
        metrics=metrics, reduce_fn=reduce_fn
    )

def evaluate_on_experts_kfolds(
    rank_fns: Sequence[Callable], fold_inds: Sequence[Tuple[int, int]], groups: np.ndarray,
    features: Dict[str, Callable], top_n: int,
    metrics: List[str] = ["ndcg", "dcg"], reduce_fn: Callable | None = np.mean
) -> Dict:
    """
    NOTE: this function is not optimized at all, as the evaluation
    design is not built with k-fold cross validation in minds.
    However, only 1 model uses it, so it's fine.
    """

    scores = defaultdict(lambda: [])

    for (query_start, query_end), rank_fn in zip(fold_inds, rank_fns):
        res = evaluate(
            rank_fn, features, ("label", None), top_n,
            data_name="annotations", data_split="test", batch_size=512,
            metrics=metrics, reduce_fn=None
        )

        start = np.sum(groups[:query_start])
        end = start + np.sum(groups[query_start:query_end])
        for key, arr in res.items():
            scores[key].extend(arr[start:end])
    
    if reduce_fn is None:
        return scores
    else:
        return { metric: reduce_fn(score) for metric, score in scores.items() }

def main():
    # deadlock may occur because pytorch dataloader and jax are used, but fine if not at the same time
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="multiprocessing.popen_fork", lineno=66)

    if len(sys.argv) <= 2:
        print("Missing config and/or params arguments!")
        return
    
    config, config_path = load_config(sys.argv[1])
    print(f"Loaded config from {config_path}")
    params_dir = sys.argv[2]
    print(f"Loading params from {params_dir}")

    # determine on which dataset is evaluated
    if config["ground_truth"] == "logging_policy":
        eval_fn = evaluate_on_logging_policy
    elif config["ground_truth"] == "experts":
        eval_fn = evaluate_on_experts
    else:
        raise Exception(f"Invalid ground truth {config['ground_truth']}")
    print(f"Evaluating on {config['ground_truth']}...")

    score_df = pd.DataFrame()

    # evaluate models
    for config_file in [config["lpe"]] + config["eval_rankers"]:
        ranker_config, ranker_config_dir = load_config(config_file)
        print(f"Loaded config from {ranker_config_dir}")
        params_path = load_param_path(params_dir, ranker_config)
        print(f"Params path: {params_path}")

        # check if model is trained
        if ranker_config["name"] != "random" and not os.path.isfile(params_path) and not os.path.isdir(params_path):
            print(f"{ranker_config['name']} has not been trained! Skipping...")
            continue

        print(f"Evaluating {ranker_config['name']}...")
        
        if ranker_config["name"] == "lambdamart":
            ranker = lambdamart.load(params_path, ranker_config)
            scores = eval_fn(
                ranker.predict, ranker.features, TOP_N, metrics=[METRIC], reduce_fn=None
            )
        elif ranker_config["name"] == "lambdamart_expert":
            rankers = []
            for i, ranker in enumerate(range(ranker_config["k-folds"])):
                rankers.append(lambdamart.load(f"{params_path}/{i}.txt", ranker_config))
            
            # TODO: fix this hot mess of a mess
            test_loader = data.load_dataloader("annotations", "test", batch_size=-1, labels={"label": None}, pad=False)
            groups = next(iter(test_loader))["groups"]

            if config["ground_truth"] == "experts":
                scores = evaluate_on_experts_kfolds(
                    [ranker.predict for ranker in rankers],
                    lambdamart.fold_iter(len(groups), ranker_config["k-folds"]),
                    groups, rankers[0].features, TOP_N,
                    metrics=[METRIC], reduce_fn=None
                )
            else:
                raise Exception("TODO")
        elif ranker_config["name"] == "upe_logging_policy":
            ranker, params = upe.load_logging_policy(params_path, ranker_config)
            scores = eval_fn(
                partial(ranker.apply, params, training=False), ranker.features, TOP_N,
                metrics=[METRIC], reduce_fn=None
            )
        # this accounts for all two tower based models
        elif ranker_config["name"] == "two_towers":
            ranker, params = two_towers.load(params_path, ranker_config)

            scores = eval_fn(
                partial(ranker.relevance.apply, { "params": params["params"]["relevance"] }),
                ranker.relevance.features, TOP_N, metrics=[METRIC], reduce_fn=None
            )
        elif ranker_config["name"] == "naive":
            ranker, params = naive.load(params_path, ranker_config)

            scores = eval_fn(
                partial(ranker.apply, params), ranker.features, TOP_N,
                metrics=[METRIC], reduce_fn=None
            )
        elif ranker_config["name"] == "random":
            ranker = random.RandomRanker()
            scores = eval_fn(
                ranker.predict, ranker.features, TOP_N,
                metrics=[METRIC], reduce_fn=None
            )
        else:
            print(f"Model {ranker_config['name']} not recognized! Skipping...")
            continue
        
        # TODO: really use config file as name?
        score_df.insert(len(score_df.columns), config_file, scores[METRIC])

    # config file of logging policy estimate model
    lpe_ranker = config["lpe"]
    # discretize logging policy quality
    score_df["logging_policy_bucket"] = pd.cut(score_df[lpe_ranker], bins=config["buckets"])
    score_df.to_csv(f"../res/eval/eval_on_{config['ground_truth']}.csv")

    # TODO: can't I do this whole shazam in the jupyter notebook (including bucketing)?
    # average scores per bucket
    #means_df = score_df.groupby("logging_policy_bucket").mean()
    # TODO: add lpe_name, so it's index and in columns
    #print(means_df) # TODO: include number of values per logging policy? (maybe in count_df)
    #means_df.to_csv(f"../res/eval/eval_on_{config['ground_truth']}.csv")

if __name__ == "__main__":
    main()