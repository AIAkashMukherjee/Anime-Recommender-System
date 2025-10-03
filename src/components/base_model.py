import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation,BatchNormalization,Input,Embedding,Dot,Dense,Flatten
from src.logger.log import get_logger
from src.exception.exception_handler import CustomException
from config.path_config import *
from utils.common_function import read_yaml
import os ,sys

logger = get_logger(__name__)

class BaseModel:
    def __init__(self,config_path):
        try:
            self.config = read_yaml(config_path)
            logger.info("Loaded configuration from config.yaml")
        except Exception as e:
            raise CustomException("Error loading configuration",sys)
        
    def Recommend(self,n_users,n_anime):
        try:
            embedding_Size=self.config["model"]["embedding_size"]

            user=Input(name='user',shape=[1])
            user_embedding = Embedding(name="user_embedding",input_dim=n_users,output_dim=embedding_Size)(user)

            anime=Input(name='anime',shape=[1])
            anime_Embedding=Embedding(name="anime_embedding",input_dim=n_anime,output_dim=embedding_Size)(anime)

            x = Dot(normalize=True , axes=2)([user_embedding,anime_Embedding])

            x = Flatten()(x)

            x = Dense(1,kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation("sigmoid")(x)

            model = Model(inputs=[user,anime], outputs=x)
            model.compile(
            loss = self.config["model"]["loss"],
            optimizer = self.config["model"]["optimizer"],
            metrics = self.config["model"]["metrics"]
            )
            return model    
    
        except Exception as e:
            raise CustomException(e,sys)
        