import src.file_names as file_names
from src.reports import prepere_xls_reports
import time
from src.uber_model import UberClassificationHandler
import src.text_parsing as tp
import time
import os
import pandas as pd
import collections
from pathlib import Path
from ast import literal_eval

#Init to create folders to store classifiers, pca or vectorization models, results, etc.
def init_folders():
    Path("local").mkdir(exist_ok=True)
    Path(file_names.best_classifiers_dir).mkdir(parents=True, exist_ok=True)
    Path(file_names.best_results_dir).mkdir(parents=True, exist_ok=True)
    Path(file_names.uber_results_dir).mkdir(parents=True, exist_ok=True)
    Path(file_names.log_dir).mkdir(parents=True, exist_ok=True)
    Path("local/datasets").mkdir(parents=True, exist_ok=True)
    Path(file_names.optuna_study_uber_dir).mkdir(parents=True, exist_ok=True)

class Arxiv:
    """
    Main class to rule them all. 
    There is combined every step from our pipeline. 
    This class call basic and uber training, dataframe creation, parsing.
    """
    def __init__(self) -> None:
        self.__dataframe = None

    def prepare_dataframe(self) -> None:
        start_time = time.time()
        if os.path.isfile(file_names.df_parsed):
            self.__dataframe = pd.read_csv(file_names.df_parsed,
                                            index_col=0,
                                            keep_default_na=False,
                                            converters={
                                                "words_alpha": eval,
                                                "sents_alpha": eval,
                                                "words_stem": eval,
                                                "words_lemma": eval,
                                                }
                                            )
                                 
        else:
            df = pd.read_csv(file_names.df)
            self.__dataframe = self.parse_dataset(df)
        print("Prepare data in  %s seconds ---" % (time.time() - start_time))

    def parse_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add new columns from parsing to dataframe. 
        Parameters:
        -----------
        df - dataframe containing 3 columns: [terms, abstract, title]

        Returns:
        --------
        df_parsed - parsed df with additional columns:
            - words aplha
            - sents alpha
            - words lemma
            - words stem
            - One Hot Encoding columns for every class
        """

        df_parsed = tp.add_parsed_columns(df)
        df_parsed = self._add_binary_columns(df_parsed)
        self._save_data(file_names.df_parsed, df_parsed)
        return df_parsed

    
    def _add_binary_columns(self, df_parsed: pd.DataFrame) -> pd.DataFrame:
        """
        Create separate column per each class
        """
        df_parsed["terms"] = df_parsed["terms"].apply(lambda x: literal_eval(x)) #Change list stored as string to list type
        counter = df_parsed["terms"].apply(collections.Counter)
        class_df = (
            pd.DataFrame.from_records(counter).fillna(value=0).astype(float)
        )
        cols = [c for c in class_df.columns if c is not None]
        class_df = class_df[cols]
        binary_class_df = class_df.applymap(lambda x: 1 if x > 0 else 0)
        df_parsed = pd.concat([df_parsed, binary_class_df], axis=1)
        return df_parsed


    def _save_data(self, file_path: str, df: pd.DataFrame) -> None:
        """
        Create or replace .csv file
        e.g. df_parsed
        """
        if os.path.isfile(file_path):
            os.remove(file_path)
        df.to_csv(file_path)


    def train_uber_model(self, classes: list, input_column: list) -> None:

        """
        Method to call uber classification. 
        """
        uber_model = UberClassificationHandler(
            self.__dataframe,
            classes,
            input_column
        )
        uber_model.train_uber_classifier()

if __name__ == "__main__":
    start_time = time.time()
    init_folders()
    arxiv = Arxiv()
    arxiv.prepare_dataframe()  #Dataframe preparation (adding columns with lemmantization, stemming and binary columns for objective labels)
    # Now we store values and infration what parameters we want to check in calculations for classifiers
    classes = ['stat.ML', 'stat.TH', 'stat.CO', 'stat.ME', 'stat.OT']  #define for which class we want to create models
    input_column = ["words_stem", "words_lemma",]  #define which methods we want to use from text processing  

    arxiv.train_uber_model(classes, input_column)

    # Convert csv files with result to xls files with worksheets as component areas
    prepere_xls_reports()
    print("Script completed in {} seconds ---".format((time.time() - start_time)))
