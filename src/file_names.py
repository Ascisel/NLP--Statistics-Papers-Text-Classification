import glob
"""
Folders in which we store models, results, training and testing files etc. 
We use it to make computation faster and instead of fit something twice just load existing one.
"""
best_dir = "output/best"
best_classifiers_dir = "output/best/classifiers"
best_results_dir = "output/best/results"
log_dir = "output/log"
features_id_dir = "output/features_id/random_split"
uber_results_dir = "{}/uber".format(best_results_dir)
best_vectorizers_dir = "{}/vectorizers".format(best_dir)
best_classifiers_dir_uber = f"{best_classifiers_dir}/uber"
optuna_study_dir = f"{best_dir}/studies"
optuna_study_uber_dir = f"{optuna_study_dir}/uber"
report_uber_summary = "{}/report_uber.xlsx".format(uber_results_dir)
best_report_summary = "{}/report.xlsx".format(best_results_dir)
local_data_set = "local/datasets"
df = "{0}/arxiv_data.csv".format(local_data_set)
df_parsed = "{0}/arxiv_data_parsed.csv".format(local_data_set)


def x_train_file(study_field, cp):
    return f"{best_vectorizers_dir}/X_vec_train_{study_field}_{cp.name}"


def x_test_file(study_field, cp):
    return f"{best_vectorizers_dir}/X_vec_test_{study_field}_{cp.name}"


def best_model_file(study_field, cp):
    return f"{best_classifiers_dir}/cm_{study_field}_{cp.name}"


def blender_model(study_field):
    return glob.glob(f"{best_classifiers_dir_uber}/{study_field}_KERAS_model_*.h5")[
        0
    ]


def best_results_file(study_field):
    return f"{best_results_dir}/{study_field}.csv"


def uber_results(study_field):
    return "{}/{}.csv".format(uber_results_dir, study_field)
