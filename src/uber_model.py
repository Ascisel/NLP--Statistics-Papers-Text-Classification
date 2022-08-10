import src.file_names as file_names
from joblib import load
import numpy as np
from sklearn.metrics import confusion_matrix
from multiprocessing import Process
import pandas as pd
from tensorflow import keras
from tensorflow.keras import Sequential
import tensorflow as tf
import shutil
from src.metrics import NewMetricCallback, new_metric
from src.weights import get_weights
from src.seq_info import words_counter, get_mean_seq_length
from typing import Union
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# optuna
import optuna
import optkeras.optkeras as opk

#optuna settings
opk.get_trial_default = lambda: optuna.trial.FrozenTrial(
    None, None, None, None, None, None, None, None, None, None, None
)
from optkeras.optkeras import OptKeras

#thread settings for keras
tf.config.threading.set_intra_op_parallelism_threads(1)
#turn off warnings
import warnings
warnings.filterwarnings("ignore")
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.enable_eager_execution()

class UberClassificationHandler:
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        classes: list,
        input_column: list
    ) -> None:
        self.__completed_flag = {}
        self.classes = classes
        self.__dataframe = dataframe
        self.input_column = input_column

    def train_uber_classifier(self) -> None:
        """
        same thread parallelism as in basic classification
        """
        threads = []
        for cl in self.classes:
            self.__completed_flag[cl] = False
            t = Process(
                target=self.build_and_check_uber_models,
                args=(cl,),
                name=cl,
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def build_and_check_uber_models(
        self, cl: str) -> None:
        print(f"{cl} COMP STARTED.")
        study_name = cl + "_KERAS"
        self.ok = OptKeras(
            directory_path=file_names.best_classifiers_dir_uber,
            sampler=optuna.samplers.TPESampler(
                consider_prior=True,
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=False,
                n_startup_trials=10,
                n_ei_candidates=24,
                seed=None,
            ),  #sampler for pruner
            pruner=optuna.pruners.SuccessiveHalvingPruner(
                min_resource=5, reduction_factor=4, min_early_stopping_rate=0
            ),  #Pruner to cut unpromising trials
            storage=f"sqlite:///{file_names.optuna_study_uber_dir}/{cl}.db",  #storage place to save studies
            load_if_exists=True,
            # parameters for OptKeras
            enable_pruning=True,
            enable_optuna_log=True,
            enable_keras_log=False,
            models_to_keep=1,
            save_best_only=True,
            verbose=0,
            monitor="new_metric",  #new_metric as default metric for trials
            direction="maximize",
            study_name=study_name,
        )

        self.ok.optimize(
            lambda trial: self.objective(trial, cl),
            n_trials=20,
            n_jobs=1,
            gc_after_trial=True,
        )
        # load file with best uber results
        df = pd.read_csv(self.ok.optuna_log_file_path)
        cleaned_sorted_df = self.__sort_and_clean_df(df, sort_by="new_metric")
        cleaned_sorted_df.to_csv(self.ok.optuna_log_file_path)
        shutil.move(
            self.ok.optuna_log_file_path, file_names.uber_results(cl)  #move csv file to specific folder 
        )

    def objective(
        self, trial: optuna.Trial, cl: str, 
    ) -> Union[float, np.float64]:
        

        # Splitting dataframe for train and test
        DF = self.__dataframe
        input_column = trial.suggest_categorical("input_column", self.input_column)
        column = input_column 
        X = DF.loc[:,column]  #Choose which column to choose. In this case we want only stem and lemma which we defined in __main__ file
        Y = DF.loc[:,cl]  #Choose column for only this (specifed earlier) class. e.g. only for stat.ML. 
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=0, stratify=Y
        )

        # ---------PARAMS FOR LSTM--------------#
        counter = words_counter(x_train)
        n_unique_words = len(counter)


        x_train_sen = x_train.to_numpy()
        x_test_sen = x_test.to_numpy()

    
        # --------TOKENIZE DATA-----------------#
        tokenizer = Tokenizer(num_words=n_unique_words)
        tokenizer.fit_on_texts(x_train_sen)

        x_train_seq = tokenizer.texts_to_sequences(x_train_sen)
        x_test_seq = tokenizer.texts_to_sequences(x_test_sen)

        # -------PADDING------------------------#
        max_length = int(get_mean_seq_length(x_train_seq))

        x_train_pad = pad_sequences(x_train_seq, maxlen=max_length, padding='post', truncating='post')
        x_test_pad = pad_sequences(x_test_seq, maxlen=max_length, padding='post', truncating='post')

        
        #---------HYPERPARAMS FOR KERAS--------#
        input_shape = max_length
        weights = get_weights(y_train)
        trial.set_user_attr('weight', weights)
        init_lr = trial.suggest_uniform("init_lr", 0.0001, 0.01)
        activation_function = trial.suggest_categorical(
            "activation_function", ["relu", "elu"]
        )
        n_hidden = trial.suggest_int('n_hidden', 2, 64)
        n_cells = trial.suggest_int('n_cells', 2, 64)
        epochs = trial.suggest_int('n_epochs', 2, 15)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 28, 64, 128])

        blend_classifier = self.__build_model(
            embedding_dim=trial.suggest_int("embedding_dim", 16, 128, 16),
            activation_function=activation_function,
            n_hidden=n_hidden,
            n_cells=n_cells,
            init_lr=init_lr,
            kernel_initializer="he_normal",
            input_shape=input_shape,
            n_unique_words=n_unique_words,
        )

        blend_classifier.fit(
            x_train_pad,
            y_train,
            class_weight={0: weights[0], 1: weights[1]},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test_pad, y_test),
            callbacks=[
                NewMetricCallback((x_test_pad, y_test)),  # set new_metric to callbacks
                self.ok.callbacks(trial),
            ],
            verbose=self.ok.keras_verbose,
        )

        #-------------SET USER PARAMS--------------#
        prd_proba = blend_classifier.predict(x_test_pad)
        prd_test = np.rint(prd_proba) # treshold
        cm = confusion_matrix(y_test, prd_test)
        trial.set_user_attr("cm", str(cm))
        prd_proba_train = blend_classifier.predict(x_train_pad)
        train_metric = new_metric(y_train, prd_proba_train)
        trial.set_user_attr("train_score", round(train_metric, 5))
        return self.ok.trial_best_value

    def __build_model(
        self,
        embedding_dim: int,
        activation_function: str,
        n_hidden: int,
        n_cells: int,
        init_lr: float,
        kernel_initializer: str,
        input_shape: int,
        n_unique_words: int,

    ) -> Sequential:
        """
        Build a sequential network with keras API
        Params:
        ----------
        embedding_dim - dimension of embedding layer (16 -> 128)
        activation_function - activation function (relu, elu)
        n_hidden - number of neurons in dense layer ;) (2 -> 64)
        n_cells - number of cells in LSTM layer (4 -> 128)
        init_lr - learning rate (0.0001 -> 0.1)
        kernel_initializer - initzializer (he_normal)
        input_shape - shape of data
        n_unique_words = number of unique words occurences in training data
        """
        model = keras.models.Sequential()
        
        model.add(keras.layers.Embedding(n_unique_words, 
                                        embedding_dim,
                                        input_length=input_shape)
        )

        model.add(keras.layers.Bidirectional(
          keras.layers.LSTM(n_cells,
                      kernel_initializer=kernel_initializer,
                      recurrent_regularizer=keras.regularizers.l2(0.01)
                      )
        )
        )

        model.add(keras.layers.Dense(
                n_hidden,
                kernel_regularizer=keras.regularizers.l2(0.01),
                activation=activation_function
        )
        )

        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(1, activation="sigmoid"))

    
        optimizer = tf.keras.optimizers.Nadam(learning_rate=init_lr, beta_1=0.9, beta_2=0.999)

        model.compile(loss="binary_crossentropy", optimizer=optimizer, run_eagerly=True)
        return model

    def __sort_and_clean_df(self, df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
        """
        clean dataset to make results more readable
        """
        df = df.rename(columns=lambda x: x.replace("__", " "))
        df = df.rename(columns=lambda x: x.replace("_", ""))
        df = df.rename(columns=lambda x: x.replace("user attrs", ""))
        df = df.rename(columns=lambda x: x.replace("params", ""))
        df = df.rename(columns=lambda x: x[1:] if x.startswith(" ") else x)
        df = df.rename(columns=lambda x: x.replace(" ", "_"))
        df = df[df.state == "COMPLETE"]
        columns_to_save = [
            "activation_function",
            "embedding_dim",
            "batch_size",
            "init_lr",
            "n_epochs",
            "n_hidden",
            "n_cells",
            "weight",
            "new_metric",
            "train_score",
            "cm",
            "val_loss",
            "datetime_start",
            "datetime_complete",
            "duration",
        ]
        cleaned_df = df[columns_to_save]
        cleaned_and_sorted_df = cleaned_df.sort_values(sort_by, ascending=False)
        return cleaned_and_sorted_df
