import torch
import pytorch_lightning as pl

from typing import Dict, Optional
from torch.nn import functional as F
from sentence_transformers.models import Pooling
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModelForSequenceClassification


class SequenceClassification(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        num_labels: int, 
        backbone_lr: float = 2e-5,
        task_head_lr: float = 2e-4,
        weight_decay: float = 1e-4,
        id2label: Optional[Dict[int, str]] = None
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            num_labels (int): Number of labels.
            backbone_lr (float, optional): The LLM's learning rate.
            Defaults to 2e-5.
            task_head_lr (float, optional): Task head/s' learning rate.
            Defaults to 2e-4.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
        self.backbone_lr = backbone_lr
        self.task_head_lr = task_head_lr
        self.weight_decay = weight_decay
        self.num_labels = num_labels

        # For metric reporting purposes
        self.id2label = id2label

        # Base model        
        self.seq_classifier = AutoModelForSequenceClassification.from_pretrained(
            mpath, num_labels=num_labels
        )

        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if not self.seq_classifier.config.pad_token_id:
            self.seq_classifier.config.pad_token_id = \
                self.seq_classifier.config.eos_token_id

        # TO-DO: Freeze the backbone model
        # if not backbone_lr:
        #     for param in self.seq_classifier.parameters():
        #         param.requires_grad = False

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, text_tokens, targets=None):
        # Push all inputs to the device in use
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
        if targets is not None:
            targets = targets.to(self.device)

        return self.seq_classifier(**text_tokens, labels=targets)

    def common_step(self, batch, batch_idx):
        ids, text_tokens, targets = batch
        output = self(text_tokens, targets)

        # Compute the metrics
        granular_metric = MulticlassF1Score(num_classes=self.num_labels, average=None).to(self.device)
        overall_metric = MulticlassF1Score(num_classes=self.num_labels, average='micro').to(self.device)
        macro_metric = MulticlassF1Score(num_classes=self.num_labels).to(self.device)
        reshaped_targets = targets.unsqueeze(1).to(self.device)
        pred_labels = output.logits.argmax(dim=1).unsqueeze(1)

        return {
            'loss': output.loss,
            'overall_f1_score': overall_metric(
                pred_labels,
                reshaped_targets
            ),
            'macro_f1_score': macro_metric(
                pred_labels,
                reshaped_targets
            ),
            'granular_f1_score': {
                self.id2label[idx]: f1 for idx, f1 in enumerate(granular_metric(
                    pred_labels,
                    reshaped_targets
                ))
            } if self.id2label is not None else granular_metric(
                pred_labels,
                reshaped_targets
            )
        }

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k, v in loss_dict.items():
            if isinstance(v, dict): continue
            self.log("train_" + k, v.item(), prog_bar=True)

        return loss_dict

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k, v in loss_dict.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    self.log(f"val_{k}_{subk}", subv.item(), prog_bar=True)
            else:
                self.log("val_" + k, v.item(), prog_bar=True)

        return loss_dict

    def test_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k, v in loss_dict.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    self.log(f"test_{k}_{subk}", subv.item(), prog_bar=True)
            else:
                self.log("test_" + k, v.item(), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        ids, text_tokens = batch

        with torch.no_grad():
            output = self(text_tokens)
            confidence_scores = output.logits.softmax(dim=1)

        return (
            ids,
            output.logits.argmax(dim=1).cpu(),
            confidence_scores.cpu()
        )

    def configure_optimizers(self):
        param_dicts = [
            {"params": self.parameters()}
            # TO-DO: Implement different LRs for backbone and task head
            # {"params": self.task_head.parameters()},
            # {"params": self.interaction_model.parameters()},
            # {
            #     "params": self.seq_classifier.parameters(),
            #     "lr": self.backbone_lr,
            # },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.task_head_lr,
            weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=4),
                "monitor": "val_loss"
            },
        }

