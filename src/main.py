# pylint: disable=import-error
import os
from transformers import TrainingArguments
import wandb
import torch

from data import GlaucomaDataset
from trainer import WeightedTrainer, get_model

wandb.login()

hyperparams = {
    "RESIZE": (2000, 2000),
    "YOLO-SIZE": (640, 640),
    "TARGET-SIZE": (512, 512),

    "YOLO-PATH": "models/yolo.torchscript",
    "SAVE-PATH": "data/preprocessed",

    "VAL-SPLIT": 0.1,
    "TEST-SPLIT": 0.1,

    "NUM-LABELS": 1, # Binary Classification
    "SEED": 9,
    "EPOCHS": 10
}

hf_params = {
    "problem_type": "single_label_classification",
    "num_labels": 1,
    "ignore_mismatched_sizes": True
    # "attention_probs_dropout_prob": 0.1,
    # "hidden_dropout_prob": 0.1,
    # "window_size": 7 # For SWIN if needed
}

models = {
    "ViT": ('google/vit-base-patch16-224-in21k', 64),
    "DeiT": ("facebook/deit-base-distilled-patch16-224", 60),
    "SWIN": ("microsoft/swin-base-patch4-window7-224", 32)
}

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])


data_path = "data/"
csv_name = "labels.csv"
csv_path = os.path.join(data_path, csv_name)

dataset = GlaucomaDataset(f"{data_path}{csv_name}", data_path, hyperparams)
pos_weight = dataset.get_pos_weight()
train, val, test = GlaucomaDataset.split(dataset, hyperparams["VAL-SPLIT"], hyperparams["TEST-SPLIT"])
train.oversample()
torch.cuda.empty_cache()

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
    num_train_epochs = hyperparams["EPOCHS"],
    dataloader_num_workers = 8,
    report_to = "wandb",
    logging_strategy = "steps",
    logging_steps = 0.1 / hyperparams["EPOCHS"],
    bf16 = True,
    seed = hyperparams["SEED"],
    label_names = ["labels"],
    label_smoothing_factor = 0.1
    # The trainer will automatically handle distributed training
)

# Use Huggingface Trainer
TEACHER_NAME = "ViT"
for name, (id, batch_size) in models.items():
    model = get_model(id, have_trained=False, **hf_params)
    training_args.output_dir = os.path.join("models", name)
    training_args.run_name = name
    teacher = None
    if name == "DeiT":
        teacher = get_model(os.path.join("models", TEACHER_NAME), have_trained=True)
    training_args.per_device_train_batch_size = batch_size
    trainer = WeightedTrainer(pos_weight, model = model, args = training_args,
                            train_dataset = train, eval_dataset = val,
                            teacher_model = teacher)

    trainer.train()
    trainer.save_model(training_args.output_dir)
    # trainer.evaluate(val)
    wandb.finish()
