from models import lambdamart, upe, two_towers, naive, random
from utils.config import load_config, load_param_path
import models.shared
import eval
from utils import data
import sys, os.path, time, warnings
from pathlib import Path
from functools import partial

# top 10 documents is optimized for
TOP_N = 10

def main():
    # deadlock may occur because pytorch dataloader and jax are used, but fine if not at the same time
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="multiprocessing.popen_fork", lineno=66)

    if len(sys.argv) <= 2:
        print("Missing config and/or params arguments!")
        return
    
    config, config_dir = load_config(sys.argv[1])
    print(f"Loaded config from {config_dir}")
    params_path = load_param_path(sys.argv[2], config)
    print(f"Params path: {params_path}")

    # check if file/directory already exists
    train = not os.path.isfile(params_path) and not os.path.isdir(params_path)

    if config["name"] != "random" and train:
        print(f"No params found, training {config['name']}")
        start_time = time.time()
    else:
        print(f"Params found, loading {config['name']} from path...")

    # TODO: consistent function naming
    # TODO: store logs
    # TODO: structure more consisely, if most functions share signature that will help
    # TODO: ^ loading features should be part of fit function, it should only take config
    # train if not yet done and evaluate
    if config["name"] == "lambdamart":
        if train:
            ranker = lambdamart.fit_logging_policy(config)
            print(f"Training took {time.time() - start_time:.1f} seconds")

            ranker.save(params_path)
        else:
            ranker = lambdamart.load(params_path, config)

        print("Evaluating...")
        print("logging policy:", eval.evaluate_on_logging_policy(ranker.predict, ranker.features, TOP_N))
        print("expert labels:", eval.evaluate_on_experts(ranker.predict, ranker.features, TOP_N))
    elif config["name"] == "lambdamart_expert":
        if train:
            rankers = lambdamart.fit_expert_labels(config)
            print(f"Training took {time.time() - start_time:.1f} seconds")

            # create params dir if it doesn't exist
            Path(params_path).mkdir(parents=True, exist_ok=True)

            for i, ranker in enumerate(rankers):
                ranker.save(f"{params_path}/{i}.txt")
        else:
            rankers = []
            for i, ranker in enumerate(range(config["k-folds"])):
                rankers.append(lambdamart.load(f"{params_path}/{i}.txt", config))
        
        print("Evaluating...")
        # TODO: fix this mess of a mess
        test_loader = data.load_dataloader("annotations", "test", batch_size=-1, labels={"label": None}, pad=False)
        groups = next(iter(test_loader))["groups"]

        print(
            "expert labels:",
                eval.evaluate_on_experts_kfolds(
                    [ranker.predict for ranker in rankers], lambdamart.fold_iter(len(groups), config["k-folds"]),
                    groups, rankers[0].features, TOP_N
                )
        )
    elif config["name"] == "upe_logging_policy":
        if train:
            ranker, params = upe.fit_logging_policy(config)
            print(f"Training took {time.time() - start_time:.1f} seconds")

            models.shared.save_flax_params(params, params_path)
        else:
            ranker, params = upe.load_logging_policy(params_path, config)

        print("Evaluating...")
        rank_fn = partial(ranker.apply, params, training=False)
        print("logging policy:", eval.evaluate_on_logging_policy(rank_fn, ranker.features, TOP_N))
        print("expert labels:", eval.evaluate_on_experts(rank_fn, ranker.features, TOP_N))
    elif config["name"] == "upe":
        # TODO: retrain
        if train:
            ranker, params = upe.fit(config)
            print(f"Training took {time.time() - start_time:.1f} seconds")

            models.shared.save_flax_params(params, params_path)
        else:
            ranker, params = upe.load(params_path, config)
        
        print("Evaluating...\nTODO: make something up here lol")
    # this accounts for all two tower based models
    elif config["name"] == "two_towers":
        if train:
            ranker, params = two_towers.fit(config)
            print(f"Training took {time.time() - start_time:.1f} seconds")

            models.shared.save_flax_params(params, params_path)
        else:
            ranker, params = two_towers.load(params_path, config)

        print("Evaluating...")
        print("expert labels:", eval.evaluate_on_experts(
            partial(ranker.relevance.apply, { "params": params["params"]["relevance"] }),
            ranker.relevance.features, TOP_N
        ))
    elif config["name"] == "naive":
        if train:
            ranker, params = naive.fit(config)
            print(f"Training took {time.time() - start_time:.1f} seconds")

            models.shared.save_flax_params(params, params_path)
        else:
            ranker, params = naive.load(params_path, config)

        print("Evaluating...")
        print("expert labels:", eval.evaluate_on_experts(
            partial(ranker.apply, params), ranker.features, TOP_N
        ))
    elif config["name"] == "random":
        ranker = random.RandomRanker()

        print("Evaluating")
        print("logging policy:", eval.evaluate_on_logging_policy(ranker.predict, ranker.features, TOP_N))
        print("expert labels:", eval.evaluate_on_experts(ranker.predict, ranker.features, TOP_N))
    else:
        print(f"Model {config['name']} not recognized")


if __name__ == "__main__":
    main()