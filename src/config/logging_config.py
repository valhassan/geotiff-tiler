import logging.config
from datetime import datetime
from pathlib import Path

import yaml

script_dir = Path(__file__).resolve().parent
CONFIG_DIR = script_dir / "log_config.yaml"
LOG_DIR = Path.cwd().joinpath("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H_%M_%S")
logfilename =  f"{LOG_DIR}/{timestamp}.log"

with open(CONFIG_DIR, "r") as f:
    config = yaml.safe_load(f.read())
    config["handlers"]["file"]["filename"] = logfilename    
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)