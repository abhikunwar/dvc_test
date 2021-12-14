from src.utils import all_utils
import argparse
import pandas as pd
import os

def get_data(config_path):
    config = all_utils.read_yaml(config_path)
    remote_data_path = config['data_source']
    df = pd.read_csv(remote_data_path,sep = ";")
    print(df.head())
    # save dataset in the local directory
    #create path to directory: artifacts/raw_local_dir/raw_local_files
    artifacts_dir = config['artifacts']['artifacts_dir']
    raw_local_dir = config['artifacts']['raw_local_dir']
    raw_local_files = config['artifacts']['raw_local_files']

    raw_local_dir_path = os.path.join(artifacts_dir, raw_local_dir)
    all_utils.create_dir(dir_path = [raw_local_dir_path])

    raw_local_file_path = os.path.join(raw_local_dir_path, raw_local_files)
    
    df.to_csv(raw_local_file_path,sep = ";",index = False)
    print(raw_local_file_path)
    




if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    parsed_args = args.parse_args()
    get_data(config_path = parsed_args.config)
    