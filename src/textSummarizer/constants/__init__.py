#to return config and params yaml files..instead of harcoding the path to these file, these will be the constant paths that we will fill here

from pathlib import Path

CONFIG_FILE_PATH = Path('config/config.yaml')
PARAMS_FILE_PATH = Path("params.yaml")