import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from utils.common_function import read_yaml
from config.path_config import *
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__=="__main__":
    # data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    # data_ingestion.run()
    
    data_processor = DataTransformation(ANIMELIST_CSV,PROCESSED_DIR)
    data_processor.run()