import yaml
import os

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content    

def create_dir(dir_path:list):
    for path in dir_path:
        os.makedirs(path,exist_ok=True)
        print(f"directory has been created at  {path}")

def save_local_data(data,data_path):
    data.to_csv(data_path,index = False)
