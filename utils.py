import torch
import os
import random
import numpy as np
import pandas as pd
import cv2
import json


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_image(path, size=(112,112)):
    image = cv2.imread(path, 0)
    if image is None:
        return np.zeros((112, 112))
    
    image = cv2.resize(image, size) / 255
    return image.astype('f')

def get_settings(path):
    with open(path,'r') as f:
        settings = json.load(f)
    return settings

class LossMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        # incremental update
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg

def save_nested_dict_to_csv(metrics_dict, csv_path):
    """
    Flattens a nested dictionary of metrics and saves it to a CSV file.

    Parameters:
    - metrics_dict: Nested dictionary containing metric values.
    - csv_path: Path to save the CSV file.
    """
    rows = []
    
    for fold, datasets in metrics_dict.items():
        for dataset_type, metrics in datasets.items():
            for metric_name, values in metrics.items():
                for epoch, value in enumerate(values, start=1):
                    rows.append({
                        'fold': fold,
                        'dataset': dataset_type,
                        'epoch': epoch,
                        'metric': metric_name,
                        'value': value
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

def update_metrics(metrics, fold, dataset_type, metric_name, value):
    if fold not in metrics:
        metrics[fold] = {}
    
    if dataset_type not in metrics[fold]:
        metrics[fold][dataset_type] = {}
    
    if metric_name not in metrics[fold][dataset_type]:
        metrics[fold][dataset_type][metric_name] = []

    metrics[fold][dataset_type][metric_name].append(value)

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


def save_metrics_to_json(metrics, model_name, encoder=TensorEncoder):
    base_dir = './plots'
    save_path = os.path.join(base_dir, model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    filename = f"metrics_{model_name}.json"
    full_path = os.path.join(save_path, filename)
    with open(full_path, "w") as file:
        json.dump(metrics, file, cls=encoder)
    
    print(f'Saving {filename}')
    return full_path