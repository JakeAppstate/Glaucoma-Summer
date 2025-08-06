# pylint: disable=(import-error, c-extension-no-member)
import os
from transformers import TrainingArguments
import torch
import wandb

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
    "EPOCHS": 10,
    "TARGET-METRIC": "sensitivity at 95% specificity"
}

hf_params = {
    "problem_type": "single_label_classification",
    "num_labels": 1,
    "ignore_mismatched_sizes": True,
    # "attention_probs_dropout_prob": 0.1,
    # "hidden_dropout_prob": 0.1,
    # "window_size": 7 # For SWIN if needed
}

models = {
    "ViT": ('google/vit-base-patch16-224-in21k', 64),
    "DeiT": ("facebook/deit-base-distilled-patch16-224", 60),
    "SWIN": ("microsoft/swin-base-patch4-window7-224", 28)
}

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])


DATA_PATH = "data/"
CSV_NAME = "labels.csv"
csv_path = os.path.join(DATA_PATH, CSV_NAME)

dataset = GlaucomaDataset(f"{DATA_PATH}{CSV_NAME}", DATA_PATH, hyperparams)
pos_weight = dataset.get_pos_weight()
train, val, test = GlaucomaDataset.split(dataset, hyperparams["VAL-SPLIT"], hyperparams["TEST-SPLIT"])
n_total = 20_000
n_pos = int(n_total * 0.032)
train.sample(n_total, n_pos)
train.oversample()
torch.cuda.empty_cache()

MODEL_NAME = "ViT"
model_id, max_batch_size = models[MODEL_NAME]
# parameters_dict["batch_size_per_device"] = [x for x in parameters_dict["batch_size_per_device"]
#                                             if x <= max_batch_size]

TEACHER_NAME = None
TEACHER_MODEL = get_model(os.path.join("models", TEACHER_NAME), have_trained=True) \
                if TEACHER_NAME else None

# Start of wandb sweep code

sweep_config = {
    'method': 'bayes',
    'metric': {
         'name': f'eval/{hyperparams["TARGET-METRIC"]}',
         'goal': 'maximize'  
       },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3
    }
}

# hyperparameters
parameters_dict = {
    'epochs': {
        'value': 20,
        },
    'per_device_train_batch_size': {
        # 'values': [x for x in (8, 16, 32, 48, 64, 128) if x <= max_batch_size],
        'value': max_batch_size
        },
    'learning_rate': {
        'min': 1e-6,
        'max': 1e-3,
        'distribution': 'log_uniform_values'
    },
    'weight_decay': {
        'min': 1e-4,
        'max': 1e-2,
        'distribution': 'log_uniform_values'
    },
    "warmup_ratio": {
        'values': [0.0, 0.05, 0.1, 0.2]
    },
    "label_smoothing_factor": {
        'values': [0.0, 0.05, 0.1, 0.2]
        # 'value': 0.0
    },
    "hidden_dropout_prob": {
        "values": [0.0, 0.1, 0.2, 0.3]
        # 'value': 0.0
    },
    "attention_probs_dropout_prob": {
        "values": [0.0, 0.1, 0.2, 0.3]
        # 'value': 0.0
    },
    "lr_scheduler": {
        'values': ["linear", "cosine", "constant"]
    }
}

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config)

def train_model(config=None):
    torch.cuda.empty_cache()
    with wandb.init(config=config):
        config = wandb.config

        training_args = TrainingArguments(
            output_dir = os.path.join("models", MODEL_NAME),
            # overwrite_output_dir = True,
            save_strategy = "epoch",
            metric_for_best_model = f'eval_{hyperparams["TARGET-METRIC"]}',
            eval_strategy = "epoch",
            per_device_train_batch_size = config.per_device_train_batch_size,  # Per GPU batch size
            learning_rate = config.learning_rate,
            weight_decay = config.weight_decay,
            num_train_epochs = config.epochs,
            dataloader_num_workers = 8,
            report_to = "wandb",
            logging_strategy = "steps",
            logging_steps = 0.1 / config.epochs,
            bf16 = True,
            seed = hyperparams["SEED"],
            label_names = ["labels"],
            label_smoothing_factor = config.label_smoothing_factor,
            warmup_ratio = config.warmup_ratio
            # The trainer will automatically handle distributed training
        )

        hf_params["hidden_dropout_prob"] = config.hidden_dropout_prob
        hf_params["attention_probs_dropout_prob"] = config.attention_probs_dropout_prob
        model = get_model(model_id, have_trained=False, **hf_params)
        teacher = None
        trainer = WeightedTrainer(pos_weight, model = model, args = training_args,
                                train_dataset = train, eval_dataset = val,
                                teacher_model = teacher)

        trainer.train()
        # trainer.evaluate(val)
        # wandb.finish()

wandb.agent(sweep_id, train_model)
