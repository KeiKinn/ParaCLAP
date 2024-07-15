import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


class CLAP_FT_Loss(nn.Module):
    def forward(self, text_features, audio_features, logit_scale_a, label):
        labels = torch.eq(label.unsqueeze(1), label.T.unsqueeze(0)).float()
        logits_per_audio = logit_scale_a * audio_features @ text_features.T
        logits_per_text = logit_scale_a * text_features @ audio_features.T

        total_loss = (F.binary_cross_entropy_with_logits(logits_per_audio, labels) +
                      F.binary_cross_entropy_with_logits(logits_per_text, labels)) / 2
        return total_loss

class ClipLoss(nn.Module):

    def __init__(
            self,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            mlp_loss=False,
            weight_loss_kappa=0,
    ):
        super().__init__()

        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.mlp_loss = mlp_loss
        self.weighted_loss = bool(weight_loss_kappa != 0)
        self.weight_loss_kappa = weight_loss_kappa
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, text_features, audio_features, logit_scale_a, logit_scale_t=None, audio_features_mlp=None,
                text_features_mlp=None):
        device = audio_features.device
        if self.mlp_loss:
            a_logits_per_audio = logit_scale_a * audio_features @ text_features_mlp.T
            a_logits_per_text = logit_scale_a * text_features_mlp @ audio_features.T
            t_logits_per_audio = logit_scale_t * audio_features_mlp @ text_features.T
            t_logits_per_text = logit_scale_t * text_features @ audio_features_mlp.T

            # calculated ground-truth and cache if enabled
            num_logits = a_logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

            if not self.weighted_loss:
                total_loss = (
                                     F.cross_entropy(a_logits_per_audio, labels) +
                                     F.cross_entropy(a_logits_per_text, labels) +
                                     F.cross_entropy(t_logits_per_audio, labels) +
                                     F.cross_entropy(t_logits_per_text, labels)
                             ) / 4
            else:
                audio_weight = (audio_features @ audio_features.T).detach()
                audio_weight = (
                    torch.exp(torch.sum(audio_weight, axis=1) / (self.weight_loss_kappa * len(audio_weight)))).detach()
                text_weight = (text_features @ text_features.T).detach()
                text_weight = (
                    torch.exp(torch.sum(text_weight, axis=1) / (self.weight_loss_kappa * len(text_features)))).detach()
                total_loss = (
                                     F.cross_entropy(a_logits_per_audio, labels, weight=audio_weight) +
                                     F.cross_entropy(a_logits_per_text, labels, weight=audio_weight) +
                                     F.cross_entropy(t_logits_per_audio, labels, weight=text_weight) +
                                     F.cross_entropy(t_logits_per_text, labels, weight=text_weight)
                             ) / 4
        else:
            logits_per_audio = logit_scale_a * audio_features @ text_features.T
            logits_per_text = logit_scale_a * text_features @ audio_features.T

            # calculated ground-truth and cache if enabled
            num_logits = logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
            if not self.weighted_loss:
                total_loss = (
                                     F.cross_entropy(logits_per_audio, labels) +
                                     F.cross_entropy(logits_per_text, labels)
                             ) / 2
            else:
                audio_weight = (audio_features @ audio_features.T).detach()
                audio_weight = (torch.exp(
                    torch.sum(audio_weight, axis=1) / (self.weight_loss_kappa * len(audio_features)))).detach()
                text_weight = (text_features @ text_features.T).detach()
                text_weight = (torch.exp(
                    torch.sum(text_weight, axis=1) / (self.weight_loss_kappa * len(text_features)))).detach()
                total_loss = (
                                     F.cross_entropy(logits_per_audio, labels, weight=text_weight) +
                                     F.cross_entropy(logits_per_text, labels, weight=audio_weight)
                             ) / 2
        return total_loss


def get_map(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(average_precision_score(target, pred, average=None))


def get_acc(pred, target):
    pred = torch.argmax(pred, 1).numpy()
    target = torch.argmax(target, 1).numpy()
    return accuracy_score(target, pred)


def get_mauc(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(roc_auc_score(target, pred, average=None))


class LPMetrics(object):
    def __init__(self, metric_names=['map', 'acc', 'mauc']):
        self.metrics = []
        for name in metric_names:
            self.metrics.append(self.get_metric(name))
        self.metric_names = metric_names

    def get_metric(self, name):
        if name == 'map':
            return get_map
        elif name == 'acc':
            return get_acc
        elif name == 'mauc':
            return get_mauc
        else:
            raise ValueError(f'the metric should be at least one of [map, acc, mauc]')

    def evaluate_mertics(self, pred, target):
        metric_dict = {}
        for i in range(len(self.metric_names)):
            metric_dict[self.metric_names[i]] = self.metrics[i](pred, target)
        return metric_dict


def calc_celoss(pred, target):
    target = torch.argmax(target, 1).long()
    return nn.CrossEntropyLoss()(pred, target)


class LPLoss(nn.Module):

    def __init__(self, loss_name):
        super().__init__()
        if loss_name == 'bce':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif loss_name == 'ce':
            self.loss_func = calc_celoss
        elif loss_name == 'mse':
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError(f'the loss func should be at least one of [bce, ce, mse]')

    def forward(self, pred, target):
        loss = self.loss_func(pred, target)
        return loss
