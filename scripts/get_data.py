import pip
 
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])


import_or_install('PyGithub')
from github import Github

import requests

import json
import os
 
import_or_install('mlflow')
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/home/petr/project/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")
 
# Замените 'YOUR_ACCESS_TOKEN' на свой токен доступа
ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
 
# Создаем экземпляр объекта класса Github
g = Github(ACCESS_TOKEN)


with mlflow.start_run():
    repo = g.get_repo('PetrGavrilin/UsefulDatasets')
    file_content = repo.get_contents('Twitter_volume_AMZN.csv')
    download_url = file_content.download_url
    response = requests.get(download_url)

    mlflow.log_artifact(local_path="/home/petr/project/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()

with open('/home/petr/project/datasets/data.csv', 'wb') as file:
    file.write(response.content)
