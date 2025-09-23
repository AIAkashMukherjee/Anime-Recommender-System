import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from utils.common_function import read_yaml
from config.path_config import *
from src.components.data_ingestion import DataIngestion

if __name__=="__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()