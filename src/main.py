
import os
import wandb
import numpy as np
import torch
import cv2

from transformers import TrainingArguments, Trainer
from transformers import ViTForImageClassification

from data import GlaucomaDataset
from trainer import WeightedTrainer, get_model

wandb.login()

hyperparams = {
    "RESIZE": (2000, 2000),
    "YOLO-SIZE": (640, 640),
    "TARGET-SIZE": (512, 512),

    "YOLO-PATH": "models/yolo.torchscript",

    "VAL-SPLIT": 0.1,
    "TEST-SPLIT": 0.1,

    "NUM-LABELS": 1 # Binary Classification
}

hf_params = {
    "problem_type": "single_label_classification",
    "num_labels": 1
    # "attention_probs_dropout_prob": 0.1,
    # "hidden_dropout_prob": 0.1,
    # "window_size": 7 # For SWIN if needed
}

models = {
    "ViT": 'google/vit-base-patch16-224-in21k',
    "DeiT": "facebook/deit-base-distilled-patch16-224",
    "SWIN": "microsoft/swin-base-patch4-window7-224"
}


data_path = "data/"
csv_name = "labels.csv"
csv_path = os.path.join(data_path, csv_name)

print("test")
dataset = GlaucomaDataset(f"{data_path}{csv_name}", data_path, hyperparams)
pos_weight = dataset.get_pos_weight()
print("test1")
train, val, test = GlaucomaDataset.split(dataset, hyperparams["VAL-SPLIT"], hyperparams["TEST-SPLIT"])
print("test2")
train.oversample()

# print("Caching all sample in train set")
# train.load() # Cache all samples in memory
# print("Finished Caching training set")

# load datasets
# train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
# val_dataloader = DataLoader(val, batch_size=64, shuffle=True)

# data = train[0]
# img = data["pixel_values"].numpy().transpose((1, 2, 0))
# print(img.shape)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imwrite("test-img.png", img)
# raise RuntimeError("Testing Image")

# model_name = "ViT"
# model_path = 'google/vit-base-patch16-224-in21k'
# model = get_model(model_path, have_trained=False, **hf_params)

training_args = TrainingArguments(
    # output_dir = os.path.join("models", model_name),
    # overwrite_output_dir = True,
    load_best_model_at_end = True,
    save_strategy = "best",
    metric_for_best_model = "eval_loss",
    eval_strategy = "epoch",
    per_device_train_batch_size = 64,  # Per GPU batch size
    learning_rate = 1e-5,
    weight_decay = 1e-4,
    num_train_epochs = 2,
    # Might need to be 0 for caching?
    # TODO Change how Caching works
    # Either load all data before training
    # or cache as large tensor in memory
    dataloader_num_workers = 8,
    report_to = "wandb",
    logging_strategy = "steps",
    logging_first_step = True,
    logging_steps = 0.1,
    bf16 = True,
    skip_memory_metrics = False,
    dataloader_persistent_workers = True
    # The trainer will automatically handle distributed training
)

# Use Huggingface Trainer
for (name, id) in models.items():
    model = get_model(id, have_trained=False, **hf_params)
    training_args.output_dir = os.path.join("models", name)
    training_args.run_name = name
    trainer = WeightedTrainer(pos_weight, model = model, args = training_args, 
                            train_dataset = train, eval_dataset = val)

    trainer.train()
