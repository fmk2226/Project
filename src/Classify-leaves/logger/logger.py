# TODO: setup_logging(log_dir) - configure Python logging to write to both console and
#   log_dir/info.log (format with timestamp & level, config from logger_config.json or inline)
#   PrintCallback in visualization.py should use logging.getLogger(...).info() instead of print
import json
import logging.config
from pathlib import Path

def setup_logging(log_dir):
    """ Configure console and file logging for one training fold.
    
    Args:
        log_dir: Directory where this fold's info.log will be saved
    """
    log_dir=Path(log_dir)
    log_dir.mkdir(parents=True,exist_ok=True)

    config_path=Path(__file__).with_name("logger_config.json")
    with config_path.open(encoding="utf-8") as f:
        config=json.load(f)

    for handler in config['handlers'].values():
        if "filename" in handler:
            handler['filename']=str(log_dir/handler['filename'])
    logging.config.dictConfig(config)