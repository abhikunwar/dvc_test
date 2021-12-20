import pandas as pd
import argparse
import joblib
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.utils import all_utils
import os
def evaluate_model(config_data_path):
    config = all_utils.read_yaml(config_data_path)

    art_dir_name = config['artifacts']['artifacts_dir']
    model_dir_name = config['artifacts']['model_dir']
    model_file_name = config['artifacts']['model_filename']

    model_path = os.path.join(art_dir_name,model_dir_name,model_file_name)

    split_dir_name = config['artifacts']['split_data_dir']
    test_data_file_name = config['artifacts']['test_data']
    test_data_path = os.path.join(art_dir_name,split_dir_name,test_data_file_name)

    test_data = pd.read_csv(test_data_path,sep = ";")

    X_test = test_data.values[:,:-1]
    y_test = test_data.values[:,-1]

    model = joblib.load(model_path)
    predicted_value = model.predict(X_test)
    r2 = r2_score(y_test,predicted_value)
    mae = mean_absolute_error(y_test,predicted_value)
    mse = mean_squared_error(y_test,predicted_value)

    score_dir_name = config['artifacts']['score_dir']
    score_dir_path = os.path.join(art_dir_name,score_dir_name)

    all_utils.create_dir([score_dir_path])

    score_file_name = config['artifacts']['score_file_name']

    score_file_path = os.path.join(art_dir_name,score_dir_name,score_file_name)
    report = {"r2":r2,"mae":mae,"mse":mse}
    all_utils.save_report(report,score_file_path)

    

if __name__=="__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--param","-p",default="params.yaml")
    parsed_args = args.parse_args()
    # print(parsed_args.config)
    evaluate_model(parsed_args.config)
    
    
    
