import json
from pathlib import Path
from typing import Tuple, Dict

# TODO: use path builders
def load_config(file: str) -> Tuple[Dict, str]:
    """ Loads model configuration from res/models/configs/`file`.  """

    config_path = "../res/models/configs/" + file
    return json.load(open(config_path)), config_path

def load_param_path(dir: str, config: Dict) -> str:
    """ Loads model parameters path from res/models/`dir`/config["params_file"]. """

    params_dir = "../res/models/" + dir
    # create params dir if it doesn't exist
    Path(params_dir).mkdir(parents=True, exist_ok=True)
    return params_dir + "/" + config["params_file"]