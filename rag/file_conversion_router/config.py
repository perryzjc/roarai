from pathlib import Path
import configparser

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"

CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_PATH)
