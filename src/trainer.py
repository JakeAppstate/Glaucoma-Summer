# pylint: disable=import-error
from transformers import Trainer, EvalPrediction, AutoModelForImageClassification, \
                         AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import recall_score, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
# from transformers.models.deit.modeling_deit import DeiTForImageClassificationWithTeacher
import torch
import numpy as np
import wandb
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

        n = len(kwargs["train_dataset"])
        self.pred_index = 0
        self.training_preds = torch.full((n,), -1)
        self.training_labels = torch.full((n,), -1)
        self.is_training = False # Will be set to True when training starts

    # pylint: disable-next=unused-argument
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        labels = inputs.pop("labels")
        eps = self.args.label_smoothing_factor or 0
        labels_smoothed = labels.clamp(eps, 1 - eps)
        labels_smoothed = labels
        # print("labels smoothed:", labels_smoothed)
        outputs = model(interpolate_pos_encoding = True, **inputs)
        if self.teacher is not None:
            logits = torch.squeeze(outputs.logits) # average of class and distillaton
            cls_logits = torch.squeeze(outputs.cls_logits)
            dist_logits = torch.squeeze(outputs.distillation_logits)
            with torch.no_grad():
                teacher_output = self.teacher(interpolate_pos_encoding = True, **inputs)
            # logits need to be converted to prob. distribution
            teacher_logits = torch.squeeze(torch.sigmoid(teacher_output.logits))
            loss = 0.5 * loss_fun(cls_logits, labels_smoothed) + \
                   0.5 * loss_fun(dist_logits, teacher_logits)
        else:
            logits = torch.squeeze(outputs.logits)
            loss = loss_fun(logits, labels_smoothed)

        # Code for logging training metrics
        # May want to make it is own function
        self._log_training_metrics(model.training, labels, logits)
        return (loss, outputs) if return_outputs else loss

    def _log_training_metrics(self, model_training, labels, logits):
        if not model_training and self.is_training:
            # Starting evaluation
            # Compute all train metrics and log
            labels = self.training_labels.cpu().numpy()
            logits = self.training_preds.cpu().numpy()
            eval_pred = EvalPrediction(predictions = logits, label_ids=labels)
            metrics = self.compute_metrics(eval_pred)
            self.log(metrics)
            # reset predictions for next epoch
            self.training_preds = self.training_preds.fill_(-1)
            self.training_labels = self.training_labels.fill_(-1)
            self.pred_index = 0
            self.is_training = False
        elif model_training:
            self.is_training = True
            i, n = self.pred_index, len(logits)
            self.training_preds[i:i+n] = logits
            self.training_labels[i:i+n] = labels
            self.pred_index += n

    def _get_accuracy(self, predictions, labels, threshold = 0.5):
        preds = (predictions > threshold).astype(np.int32)
        acc = np.mean(preds == labels).item()
        return acc

    def _get_auc(self, predictions, labels):
        pass

    def _get_fn(self, predictions, labels, threshold = 0.5):
        predictions, labels = predictions[labels == 1], labels[labels == 1]
        preds = (predictions > threshold).astype(np.int32)
        n = np.sum(preds != labels).item()
        return n

    def _get_tn(self, predictions, labels, threshold = 0.5):
        predictions, labels = predictions[labels == 0], labels[labels == 0]
        preds = (predictions > threshold).astype(np.int32)
        n = np.sum(preds == labels).item()
        return n

    def _get_fp(self, predictions, labels, threshold = 0.5):
        predictions, labels = predictions[labels == 0], labels[labels == 0]
        preds = (predictions > threshold).astype(np.int32)
        n = np.sum(preds != labels).item()
        return n

    def _get_tp(self, predictions, labels, threshold = 0.5):
        predictions, labels = predictions[labels == 1], labels[labels == 1]
        preds = (predictions > threshold).astype(np.int32)
        n = np.sum(preds == labels).item()
        return n

    def _get_sens_at_95_spec(self, predictions, labels):
        SPECIFICITY = 0.95
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        target_fpr = 1 - SPECIFICITY
        i = _aprox_binary_search(fpr, target_fpr)
        threshold = np.squeeze(thresholds[i])

        preds = (predictions > threshold).astype(np.int32)
        sensitivity = recall_score(labels, preds)
        return sensitivity, threshold
    
    def _plot_confidence(self, preds, labels):
        prob_true, prob_pred = calibration_curve(labels, preds, n_bins=10)
        data = [[x, y] for (x, y) in zip(prob_pred, prob_true)]
        axis_names = ["Mean predicted probability", "Fraction of positives"]
        table = wandb.Table(data=data, columns=axis_names)
        wandb.log({
            "Calibration-Plot": wandb.plot.line(table, *axis_names, title = "Calibration Plot")
        })

    def _compute_metrics(self, eval_pred):
        predictions = np.squeeze(eval_pred.predictions)
        if self.teacher is not None:
            # DeiT returns 3 logits, class, distil, and average
            predictions = predictions[0] # output.logits (average)
        labels = eval_pred.label_ids
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        # convert logits to a probability distribution
        predictions = sigmoid(predictions)
        metrics = {}
        metrics["accuracy"] = self._get_accuracy(predictions, labels)
        metrics["Accuracy"] = metrics["accuracy"]
        sensitivity, threshold = self._get_sens_at_95_spec(predictions, labels)
        metrics["sensitivity at 95% specificity"] = sensitivity
        metrics["accuracy with threshold"] = self._get_accuracy(predictions, labels, threshold)
        metrics["AUC"] = roc_auc_score(labels, predictions)

        metrics["False Negative"] = self._get_fn(predictions, labels)
        metrics["False Negative with threshold"] = self._get_fn(predictions, labels, threshold)
        metrics["False Positive"] = self._get_fp(predictions, labels)
        metrics["False Positive with threshold"] = self._get_fp(predictions, labels, threshold)
        metrics["True Negative"] = self._get_tn(predictions, labels)
        metrics["True Negative with threshold"] = self._get_tn(predictions, labels, threshold)
        metrics["True Positive"] = self._get_tp(predictions, labels)
        metrics["True Positive with threshold"] = self._get_tp(predictions, labels, threshold)

        self._plot_confidence(predictions, labels)
        return metrics
    
    # def create_optimizer_and_scheduler(self, num_training_steps: int):
    #     # Your custom optimizer and scheduler creation logic here
    #     optimizer = torch.optim.AdamW(self.model.parameters(),
    #                                   lr=self.args.learning_rate,
    #                                   weight_decay = self.args.weight_decay)
    #     lr_scheduler = get_linear_schedule_with_warmup(
    #         optimizer, num_warmup_steps=self.args.num_warmup_steps,
    #         num_training_steps=num_training_steps
    #     )
    #     return optimizer, lr_scheduler


def get_model(hf_id, have_trained, **hf_params):
    if have_trained:
        model = AutoModelForImageClassification.from_pretrained(hf_id)
    else:
        config = AutoConfig.from_pretrained(hf_id, **hf_params)
        model = AutoModelForImageClassification.from_pretrained(hf_id, config = config,
                                                                ignore_mismatched_sizes=True)
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
