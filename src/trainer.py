# pylint: disable=import-error
from transformers import Trainer, AutoModelForImageClassification, AutoConfig
from sklearn.metrics import accuracy_score, recall_score, roc_curve, roc_auc_score
# from transformers.models.deit.modeling_deit import DeiTForImageClassificationWithTeacher
import torch
import numpy as np
# from torcheval.metrics.functional import binary_accuracy

class WeightedTrainer(Trainer):
    def __init__(self, pos_weight, teacher_model = None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = torch.tensor(pos_weight)
        self.teacher = teacher_model.to(self.args.device) if teacher_model is not None else None
        # self.compute_metrics is overwritten in super.init
        # even if method is overriden.
        self.compute_metrics = self._compute_metrics
        self.model_accepts_loss_kwargs = False
        
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Note: may want to set remove_unused_columns to False in TrainingArgs
        # if values aren't being passed to forward method
        loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        labels = inputs.pop("labels")
        outputs = model(interpolate_pos_encoding = True, **inputs)
        if self.teacher is not None:
            cls_logits = torch.squeeze(outputs.cls_logits)
            dist_logits = torch.squeeze(outputs.distillation_logits)
            with torch.no_grad():
                teacher_output = self.teacher(interpolate_pos_encoding = True, **inputs)
            teacher_logits = torch.squeeze(torch.sigmoid(teacher_output.logits)) # logits need to be converted to prob.
            loss = 0.5 * loss_fun(cls_logits, labels) + 0.5 * loss_fun(dist_logits, teacher_logits)
        else:
            logits = torch.squeeze(outputs.logits)
            loss = loss_fun(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
    def _get_accuracy(self, predictions, labels, threshold = 0.5):
        preds = (predictions > threshold).astype(np.int32)
        acc = np.mean(preds == labels)
        return acc
    
    def _get_auc(self, predictions, labels):
        pass

    def _get_sens_at_95_spec(self, predictions, labels):
        SPECIFICITY = 0.95
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        target_fpr = 1 - SPECIFICITY
        i = _aprox_binary_search(fpr, target_fpr)
        threshold = np.squeeze(thresholds[i])

        preds = (predictions > threshold).astype(np.int32)
        sensitivity = recall_score(labels, preds)
        return sensitivity, threshold
    
    def _compute_metrics(self, eval_pred):
        predictions = np.squeeze(eval_pred.predictions)
        print(predictions.shape)
        print(self.teacher)
        if self.teacher is not None:
            # DeiT returns 3 logits, class, distil, and average
            predictions = predictions[0] # output.logits (average)
        labels = eval_pred.label_ids
        print(labels.shape)
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        # convert logits to a probability distribution
        predictions = sigmoid(predictions)
        metrics = {}
        metrics["accuracy"] = self._get_accuracy(predictions, labels)
        sensitivity, threshold = self._get_sens_at_95_spec(predictions, labels)
        metrics["sensitivity at 95% specificity"] = sensitivity
        metrics["accuracy with threshold"] = self._get_accuracy(predictions, labels, threshold)
        metrics["AUC"] = roc_auc_score(labels, predictions)

        return metrics
        
def get_model(hf_id, have_trained, **hf_params):
    if have_trained:
        model = AutoModelForImageClassification.from_pretrained(hf_id)
    else:
        config = AutoConfig.from_pretrained(hf_id, **hf_params)
        model = AutoModelForImageClassification.from_pretrained(hf_id, config = config, ignore_mismatched_sizes=True)
    
    return model

def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

def _aprox_binary_search(array, target, left = None, right = None):
    left = left if left is not None else 0
    right = right if right is not None else len(array) - 1
    # base cases
    if left > right:
        # return closest value to target
        return left if np.abs(array[left] - target) < np.abs(array[right] - target) else right
    if left == right:
        if left == 0:
          # at beginning of array, can be 0 or 1
          right += 1
          return left if np.abs(array[left] - target) < np.abs(array[right] - target) else right
        if right == len(array) - 1:
          # at end of array, can be len(array) -1 or -2
          left -= 1
          return left if np.abs(array[left] - target) < np.abs(array[right] - target) else right
        # can be left +- 1
        return int(np.argmin(np.abs(array[left-1:left+2] - target))) + left - 1

    # regular binary search
    mid = (left + right) // 2
    if array[mid] == target:
        return mid
    elif target < array[mid]:
        right = mid - 1
    else:
        left = mid + 1
    return _aprox_binary_search(array, target, left, right)




