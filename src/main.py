from models.logging_policy import lambdamart, upe
from utils import data
import sys, os.path, json, time
from pathlib import Path

# top 10 documents is optimized for
TOP_N = 10

# percentage of data used for testing, 25% equals one partition, which is usual
TEST_DATA_PERCENT = 25

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
# bert features are normalized and require not preprocessing
BERT_FEATURE = {
    "query_document_embedding": None
}

def main():
    if len(sys.argv) <= 2:
        print("Missing config and/or params arguments!")
        return
    
    # TODO: use path builders
    config_dir = "../res/models/configs/" + sys.argv[1]
    print(f"Loading config from {config_dir}")
    config = json.load(open(config_dir))

    params_dir = "../res/models/" + sys.argv[2]
    # create params dir if it doesn't exist
    Path(params_dir).mkdir(parents=True, exist_ok=True)
    params_path = params_dir + "/" + config["params_file"]
    print(f"Params path: {params_path}")

    # check if file already exists
    train = not os.path.isfile(params_path)

    # load optional input features with corresponding preprocess functions
    features = {}
    if "ltr" in config["optional_features"]:
        features |= LTR_FEATURES
    if "bert" in config["optional_features"]:
        features |= BERT_FEATURE

    # TODO: consistent function naming
    # TODO: store logs
    # TODO: structure more consisely (do some of the printing beforehand lol)
    # train if not yet done and evaluate
    if config["name"] == "lambdamart":
        if train:
            print(f"No params found, training {config['name']}...")
            start = time.time()
            ranker = lambdamart.fit_logging_policy(features, config)
            print(f"Training took {time.time() - start:.1f} seconds")
            ranker.save(params_path)
        else:
            print(f"Params found, loading {config['name']} from path...")
            ranker = lambdamart.load_logging_policy(params_path, features)
        print("Evaluating...")
        print(lambdamart.evaluate_logging_policy(ranker, TOP_N, TEST_DATA_PERCENT))
    elif config["name"] == "upe_logging_policy":
        if train:
            print(f"No params found, training {config['name']}...")
            start = time.time()
            ranker, params = upe.fit_logging_policy(features, config)
            print(f"Training took {time.time() - start:.1f} seconds")
            upe.save_params(params, params_path)
        else:
            print(f"Params found, loading {config['name']} from path...")
            ranker, params = upe.load_logging_policy(params_path, features, config)
        print("Evaluating...")
        print(upe.evaluate_logging_policy(ranker, params, TOP_N, TEST_DATA_PERCENT))
    elif config["name"] == "upe":
        # TODO: retrain
        if train:
            print(f"No params found, training {config['name']}...")
            start = time.time()
            ranker, params = upe.fit(features, config)
            print(f"Training took {time.time() - start:.1f} seconds")
            upe.save_params(params, params_path)
        else:
            print(f"Params found, loading {config['name']} from path...")
            ranker, params = upe.load(params_path, features, config)
        print("Evaluating...")
        print(upe.evaluate(ranker, params))

if __name__ == "__main__":
    main()