import pandas as pd
import argparse
from src.utils import all_utils
import os
from sklearn.linear_model import ElasticNet
import joblib

def train_model(yaml_path,param_yaml_path):
    yaml_data = all_utils.read_yaml(yaml_path)
    yaml_param = all_utils.read_yaml(param_yaml_path)
    #print(yaml_data)
# training file name
    training_data = yaml_data['artifacts']['train_data']
    # testing file name
    testing_data = yaml_data['artifacts']['test_data']
    # training file path
    artifacts_dir = yaml_data['artifacts']['artifacts_dir']
    split_data_dir = yaml_data['artifacts']['split_data_dir']
    train = os.path.join(artifacts_dir,split_data_dir, training_data)
    # print(train)
    df = pd.read_csv(train,sep = ";")
    # print(df.head())
    # print(df.columns)
    X_train = df.values[:,:-1]
    y_train = df.values[:,-1]
    # print(X_train[0:5])
    # print(y_train[0:5])

    alpha = yaml_param['model_param']['ElasticNet']['params']['alpha']
    l1_ratio = yaml_param['model_param']['ElasticNet']['params']['l1_ratio']
    random_state = yaml_param['model_param']['ElasticNet']['params']['random_state']
    lr = ElasticNet(alpha= alpha,l1_ratio=l1_ratio,random_state=random_state)
    lr.fit(X_train,y_train)

    artifacts =  yaml_data['artifacts']['artifacts_dir']
    model_dir =   yaml_data['artifacts']['model_dir']
    model_filename = yaml_data['artifacts']['model_filename']

    model_direc = os.path.join(artifacts,model_dir)

    all_utils.create_dir([model_direc])

    model_path = os.path.join(model_direc,model_filename)
    
    joblib.dump(lr,model_path)
    # print("done")

    






if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--param","-p",default="params.yaml")
    parsed_args = args.parse_args()
    # print(parsed_args.config)
    train_model(parsed_args.config,parsed_args.param)
    

