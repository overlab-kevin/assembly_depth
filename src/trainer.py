# src/trainer.py

import sys
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from segmentation_models_pytorch.utils.meter import AverageValueMeter

class BaseEpoch:
    """
    A base class for a single epoch of training or validation.
    It handles the iteration over the dataloader and logging.
    """
    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        return ', '.join([f'{k}: {v:.4f}' for k, v in logs.items()])

    def on_epoch_start(self):
        """Actions to perform at the beginning of an epoch."""
        pass

    def batch_update(self, data):
        """The core logic for processing a single batch."""
        raise NotImplementedError

    def run(self, dataloader):
        """Iterates over the dataloader and executes the training/validation logic."""
        self.on_epoch_start()
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {m.__name__: AverageValueMeter() for m in self.metrics}
        
        # Add meters for individual loss components
        metrics_meters['loss_seg'] = AverageValueMeter()
        metrics_meters['loss_reg'] = AverageValueMeter()
        metrics_meters['loss_assembly'] = AverageValueMeter()
        metrics_meters['assembly_acc'] = AverageValueMeter()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for data in iterator:
                # Move data to the correct device
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(self.device)

                loss, y_pred, individual_losses = self.batch_update(data)

                # Update loss meter
                loss_meter.add(loss.item())
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # Update individual loss meters
                for key, value in individual_losses.items():
                    if key in metrics_meters:
                        metrics_meters[key].add(value)
                
                # Update metrics for segmentation mask
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, data["mask"]).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    iterator.set_postfix_str(self._format_logs(logs))
        
        return logs


class TrainEpoch(BaseEpoch):
    """
    An epoch for model training. Includes backpropagation and optimizer steps.
    """
    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, train_assembly=False):
        super().__init__(model, loss, metrics, 'train', device, verbose)
        self.optimizer = optimizer
        self.train_assembly = train_assembly
        self.assembly_criterion = torch.nn.BCEWithLogitsLoss()

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, data):
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_masks, pred_positions, classifications = self.model(data['image'])
        
        # Apply mask to regression output to only compute loss on foreground pixels
        for c in range(pred_positions.shape[1]):
             pred_positions[:, c, :, :] *= data['mask'][:, 0, :, :]

        # --- Calculate Loss Components ---
        # 1. Segmentation loss (from paper)
        loss_seg = self.loss(pred_masks, data['mask'])

        # 2. Point correspondence (regression) loss (from paper)
        # Mean Square Error between predicted and ground truth points
        if data['mask'].sum() > 0:
            errors = torch.norm(pred_positions[:, :3, :, :] - data['model_points_img'], dim=1)
            loss_reg = errors.sum() / data['mask'].sum()
        else:
            loss_reg = torch.tensor(0.0).to(self.device)

        # 3. Assembly classification loss (optional, from paper)
        loss_assembly = torch.tensor(0.0).to(self.device)
        assembly_acc = 0.0
        if self.train_assembly and classifications is not None:
            loss_assembly = self.assembly_criterion(classifications, data['assembly_state'])
            # Calculate accuracy for logging
            preds = classifications.detach().cpu().view(-1) > 0
            truth = data['assembly_state'].detach().cpu().view(-1) > 0
            assembly_acc = accuracy_score(truth, preds)

        # Total loss is a sum of the components
        total_loss = loss_seg + loss_reg + loss_assembly
        
        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        
        individual_losses = {
            'loss_seg': loss_seg.item(),
            'loss_reg': loss_reg.item(),
            'loss_assembly': loss_assembly.item(),
            'assembly_acc': assembly_acc,
        }

        return total_loss, pred_masks, individual_losses


class ValidationEpoch(BaseEpoch):
    """
    An epoch for model validation. No backpropagation.
    """
    def __init__(self, model, loss, metrics, device='cpu', verbose=True, train_assembly=False):
        super().__init__(model, loss, metrics, 'valid', device, verbose)
        self.train_assembly = train_assembly
        self.assembly_criterion = torch.nn.BCEWithLogitsLoss()

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, data):
        with torch.no_grad():
            # Forward pass
            pred_masks, pred_positions, classifications = self.model(data['image'])
            
            # Apply mask to regression output
            for c in range(pred_positions.shape[1]):
                 pred_positions[:, c, :, :] *= data['mask'][:, 0, :, :]

            # --- Calculate Loss Components ---
            loss_seg = self.loss(pred_masks, data['mask'])
            
            if data['mask'].sum() > 0:
                errors = torch.norm(pred_positions[:, :3, :, :] - data['model_points_img'], dim=1)
                loss_reg = errors.sum() / data['mask'].sum()
            else:
                loss_reg = torch.tensor(0.0).to(self.device)
            
            loss_assembly = torch.tensor(0.0).to(self.device)
            assembly_acc = 0.0
            if self.train_assembly and classifications is not None:
                loss_assembly = self.assembly_criterion(classifications, data['assembly_state'])
                preds = classifications.detach().cpu().view(-1) > 0
                truth = data['assembly_state'].detach().cpu().view(-1) > 0
                assembly_acc = accuracy_score(truth, preds)
                
            total_loss = loss_seg + loss_reg + loss_assembly
            
            individual_losses = {
                'loss_seg': loss_seg.item(),
                'loss_reg': loss_reg.item(),
                'loss_assembly': loss_assembly.item(),
                'assembly_acc': assembly_acc
            }

        return total_loss, pred_masks, individual_losses
