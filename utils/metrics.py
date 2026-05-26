import torch
import torch.nn as nn
from datetime import datetime
import pandas as pd
import numpy as np
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics import JaccardIndex, Accuracy, Dice, F1Score, Precision, Recall
from torchmetrics.clustering import AdjustedRandScore

# for supervised model
class get_metrics(nn.Module): 
    def __init__(self, metrics, num_classes, average, device):
        super(get_metrics, self).__init__()
        self.metrics = metrics
        self.num_classes = num_classes
        self.average = average
        self.device = device

        if num_classes ==  2:
            task = "binary"
        else:
            task = "multiclass"

        self.metrics_dict = nn.ModuleDict({})

        for idx, m in enumerate(metrics):
            if m == "IoU":
                self.metrics_dict["IoU_" + self.average[idx]] = JaccardIndex(num_classes=self.num_classes, reduction="elementwise_mean",average=self.average[idx], task=task).to(device)
                
            elif m == "accuracy":
                self.metrics_dict["accuracy_" + self.average[idx]] = Accuracy(num_classes=self.num_classes,average=self.average[idx], task=task).to(device)

            elif m == "dice":
                self.metrics_dict["dice_" + self.average[idx]] = Dice(num_classes=self.num_classes,average=self.average[idx], task=task).to(device)

            elif m == "f1_score":
                self.metrics_dict["f1_score_" + self.average[idx]] = F1Score(num_classes=self.num_classes,average=self.average[idx], task=task).to(device)

            elif m == "precision":
                self.metrics_dict["precision_" + self.average[idx]] = Precision(num_classes=self.num_classes, average=self.average[idx], task=task).to(device)

            elif m == "recall":
                self.metrics_dict["recall_" + self.average[idx]] = Recall(num_classes=self.num_classes,average=self.average[idx],task=task).to(device)

            elif m == "ari":
                self.metrics_dict["ari"] = AdjustedRandScore().to(device)


    def calculate(self, pred, gt):
        for key, metric in self.metrics_dict.items():
            if key == "ari":
                pred_ari = torch.argmax(pred, dim=1).flatten()
                gt_ari = gt.flatten()
                metric.update(pred_ari, gt_ari)
            else:
                metric.update(pred, gt)

    def compute(self):
        metrics_out = {}
        for key, metric in self.metrics_dict.items():
            metrics_out[key] = metric.compute().item()
        return metrics_out

    def reset(self):
        for key, metric in self.metrics_dict.items():
            metric.reset()

def acc_features_topk(feature, feature_aug, target, k=5):
    bs = feature.shape[0]
    cos_sim = pairwise_cosine_similarity(feature, feature_aug)
    
    topk_preds = torch.topk(cos_sim, k, dim=1).indices
    
    correct = 0
    for i in range(bs):
        if (target[i] == topk_preds[i]).any().item():
            correct += 1
    
    acc = correct / bs

    return acc * 100