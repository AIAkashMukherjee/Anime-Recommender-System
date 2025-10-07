import joblib,sys,os
import comet_ml
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,TensorBoard,EarlyStopping
from src.logger.log import get_logger
from src.exception.exception_handler import CustomException
from src.components.base_model import BaseModel
from config.path_config import *

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,data_path):
        self.data_path= data_path

        # self.experiment = comet_ml.Experiment(
        #     api_key="uqgrnGhGvBA0zC3HfdmGf2WN9",
        #     project_name="mlops-course-2",
        #     workspace="data-guru0"
        # )
        logger.info("Model Training & COMET ML initialized..")


    def load_data(self):
        try:
            X_train_array = joblib.load(X_TRAIN_ARRAY)
            X_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)

            logger.info("Data loaded sucesfully for Model Trainig")
            return X_train_array,X_test_array,y_train,y_test

        except Exception as e:
            raise CustomException(e,sys)  

    def train_model(self):   
        try:
            X_train_array,X_test_array,y_train,y_test = self.load_data()
            n_users=len(joblib.load(USER2USER_ENCODED))
            n_anime=len(joblib.load(ANIME2ANIME_ENCODED))

            base_model=BaseModel(config_path=CONFIG_PATH)
            model=base_model.Recommend(n_users=n_users,n_anime=n_anime)


            start_lr=0.0001
            min_lr=0.0001
            max_lr=0.01
            batch_Size=10000
            ramup_epochs =6
            sustain_epoch=0
            exp_decay = 0.7

            def lr_fn(epoch): #  find  best lr for our model
                if epoch<ramup_epochs:
                    return (max_lr-start_lr)/ramup_epochs*epoch + start_lr
                elif epoch<ramup_epochs+sustain_epoch:
                    return max_lr
                else:
                    return (max_lr-min_lr) * exp_decay ** (epoch-ramup_epochs-sustain_epoch)+min_lr
                
            lr_callback=LearningRateScheduler(lambda epoch:lr_fn(epoch),verbose=0)
            model_checkpoint=ModelCheckpoint(filepath=CHECKPOINT_FILE_PATH,save_weights_only=True,monitor="val_loss",mode="min",save_best_only=True)

            early_stopping = EarlyStopping(patience=3,monitor="val_loss",mode="min",restore_best_weights=True)    

            my_callbacks = [model_checkpoint,lr_callback,early_stopping]
            os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH),exist_ok=True)
            os.makedirs(MODEL_DIR,exist_ok=True)
            os.makedirs(WEIGHTS_DIR,exist_ok=True)


            history = model.fit(
                x=X_train_array,
                y=y_train,
                batch_size=batch_Size,
                epochs=20,
                verbose=1,
                validation_data = (X_test_array,y_test),
                callbacks=my_callbacks
            )
            model.load_weights(CHECKPOINT_FILE_PATH)
            logger.info("Model training Completedd.....") 

            # for epoch in range(len(history.history['loss'])):
            #     train_loss = history.history["loss"][epoch]
            #     val_loss = history.history["val_loss"][epoch]

                # self.experiment.log_metric('train_loss',train_loss,step=epoch)
                # self.experiment.log_metric('val_loss',val_loss,step=epoch)
            self.save_model_weights(model=model)
        except Exception as e:
            raise CustomException(e,sys)  
    def extract_weights(self,name,model):
        try:
            weight_layer=model.get_layer(name)
            weights = weight_layer.get_weights()[0]
            weights = weights/np.linalg.norm(weights,axis=1).reshape((-1,1))
            logger.info(f"Extracting weights for {name}")
            return weights     
        except Exception as e:
            raise CustomException(e,sys)

    def save_model_weights(self,model):
        try:
            model.save(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")

            anime_weights=self.extract_weights("anime_embedding",model)
            user_weights=self.extract_weights("user_embedding",model)

            joblib.dump(user_weights,USER_WEIGHTS_PATH)
            joblib.dump(anime_weights,ANIME_WEIGHTS_PATH)

            logger.info("User and Anime weights saved sucesfully....")

        except Exception as e:
            raise CustomException(e,sys)
            