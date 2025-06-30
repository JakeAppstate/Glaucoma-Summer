from transformers import Trainer, AutoModelForImageClassification, AutoConfig
import torch
# from torcheval.metrics.functional import binary_accuracy

class WeightedTrainer(Trainer):
    def __init__(self, pos_weight, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = torch.tensor(pos_weight)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Note: may want to set remove_unused_columns to False in TrainingArgs
        # if values aren't being passed to forward method
        loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        labels = inputs.pop("labels")
        outputs = model(interpolate_pos_encoding = True, **inputs)
        logits = torch.squeeze(outputs.logits)
        loss = loss_fun(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # acc = binary_accuracy(labels, predictions, threshold=0)
        preds = (predictions > 0).astype(torch.int)
        acc = torch.mean(preds == labels)
        metrics = {
            "Accuracy": acc
        }
        return metrics
        

def get_model(hf_id, have_trained, **hf_params):
    if have_trained:
        model = AutoModelForImageClassification.from_pretrained(hf_id)
    else:
        config = AutoConfig.from_pretrained(hf_id, **hf_params)
        print(config)
        model = AutoModelForImageClassification.from_pretrained(hf_id, config = config)
    
    return model

def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}
