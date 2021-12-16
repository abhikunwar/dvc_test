import argparse
from src.utils import all_utils
import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split



def split_data(data_path,param_path):
    content_yaml = all_utils.read_yaml(data_path)
    content_param = all_utils.read_yaml(param_path)
    #print(content_yaml)
    artifacts_dir = content_yaml['artifacts']['artifacts_dir']
    raw_local_dir = content_yaml['artifacts']['raw_local_dir']
    raw_local_files = content_yaml['artifacts']['raw_local_files']

    data_file_path = os.path.join(artifacts_dir,raw_local_dir,raw_local_files)
    print(data_file_path)
    #read the data

    df = pd.read_csv(data_file_path)
    # print(df.head())

    split_size = content_param['base']['test_size']
    random_st = content_param['base']['random_state']
    train,test = train_test_split(df,test_size=split_size,random_state=random_st)
    # print(train.size)
    # print(test.size)
    split_data_dir = content_yaml['artifacts']['split_data_dir']
    split_data_dir_path = os.path.join(artifacts_dir,split_data_dir)

    all_utils.create_dir(dir_path = [split_data_dir_path])

    train_data_path = content_yaml['artifacts']['train_data']
    train_data = os.path.join(split_data_dir_path,train_data_path)

    test_data_path = content_yaml['artifacts']['test_data']
    test_data = os.path.join(split_data_dir_path,test_data_path)

    all_utils.save_local_data(train,train_data)
    all_utils.save_local_data(test,test_data)


    

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--param","-p",default="params.yaml")

    parsed_args = args.parse_args()
    # print(parsed_args.config)
    # print(parsed_args.param)
    split_data(parsed_args.config,parsed_args.param)