import os
import pandas as pd

# pd.show_versions()

print("Before Import")
from data import GlaucomaDataset


print("starting")

hyperparams = {
    "RESIZE": (2000, 2000),
    "YOLO-SIZE": (640, 640),
    "TARGET-SIZE": (512, 512),

    "YOLO-PATH": "models/yolo.torchscript",

    "VAL-SPLIT": 0.1,
    "TEST-SPLIT": 0.1,

    "NUM-LABELS": 1, # Binary Classification
    "SEED": 9
}

data_path = "data/"
csv_name = "labels.csv"
save_dir = "preprocessed"
csv_path = os.path.join(data_path, csv_name)

dataset = GlaucomaDataset(csv_path, data_path, hyperparams)
pos_weight = dataset.get_pos_weight()
train, val, test = GlaucomaDataset.split(dataset, hyperparams["VAL-SPLIT"], hyperparams["TEST-SPLIT"])
train.oversample()

train.save(os.path.join(data_path, save_dir, "train"))
val.save(os.path.join(data_path, save_dir, "val"))