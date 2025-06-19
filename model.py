import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import lightning.pytorch as pl
import gc
import random
import torch

from datasets import get_dataset_cls, collate_fn
from models import get_model_cls
from losses import get_loss_cls
from metrics import calculate_hr_and_hrv_metrics

def get_wd_params(module):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.MultiheadAttention)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif 'time_mix' in pn:
                decay.add(fpn)
            else:
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in module.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params), )
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay)) if param_dict[pn].requires_grad]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if param_dict[pn].requires_grad], "weight_decay": 0.0},
    ]
    return optim_groups

def get_optimizer_cls(name: str):
    name = name.lower()
    if name == 'adamw':
        return optim.AdamW
    raise ValueError(f'Unknown optimizer: {name}')

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.max_epochs = config['trainer']['max_epochs']

        model_cls = get_model_cls(config['model']['name'])
        self.model = model_cls(**config['model']['hparams'])

        # prepare losses
        self.loss_names = [p['name'] for p in config['loss']]
        self.loss_weight_bases = [p['weight'] for p in config['loss']]
        self.loss_weight_exps = [p.get('exp', 1.0) for p in config['loss']]
        self.losses = nn.ModuleList([get_loss_cls(p['name'])() for p in config['loss']])
        self.matching_loss_weight = config.get('matching_loss_weight', 1.0)

    def forward(self, source_x, target_x=None):
        return self.model(source_x, target_x)

    def predict(self, x):
        return self.model.predict(x)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # Tạo iterator mới cho target mỗi epoch
        self._target_iter = iter(self.target_loader)

    def train_dataloader(self):
        # build source & target datasets
        source_sets, target_sets = [], []
        for args in self.config['data']['train_sets']:
            ds_cls = get_dataset_cls(args['name'])
            params = self.config['data']['datasets'][args['name']]
            if args.get('domain') == 'source':
                source_sets.append(ds_cls(**params,
                                           split=args['split'],
                                           split_idx=self.config.get('split_idx', 0),
                                           training=True))
            elif args.get('domain') == 'target':
                target_sets.append(ds_cls(**params,
                                           split=args['split'],
                                           split_idx=None,
                                           training=True))

        source_ds = ConcatDataset(source_sets)
        target_ds = ConcatDataset(target_sets)

        # DataLoader riêng cho target
        self.target_loader = DataLoader(
            target_ds,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.config['data']['num_workers'] > 0,
            collate_fn=collate_fn,
        )
        # DataLoader cho source
        source_loader = DataLoader(
            source_ds,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.config['data']['num_workers'] > 0,
            collate_fn=collate_fn,
        )
        return source_loader

    def training_step(self, source_batch, batch_idx):
        # Lấy batch từ target loader
        try:
            target_batch = next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(self.target_loader)
            target_batch = next(self._target_iter)

        source_frames, source_waves, source_data = source_batch
        target_frames, _, _ = target_batch

        source_frames = source_frames.to(self.device)
        source_waves = source_waves.to(self.device)
        target_frames = target_frames.to(self.device)
        
        preds, matching_loss = self(source_frames, target_frames)

        # supervised losses
        supervised_loss = 0.
        self.loss_weights = [base * (exp ** (self.current_epoch / self.max_epochs))
                             for base, exp in zip(self.loss_weight_bases, self.loss_weight_exps)]
        for name, crit, w in zip(self.loss_names, self.losses, self.loss_weights):
            l = crit(preds.squeeze(-1), source_waves)
            self.log(f'train/{name}', l, prog_bar=True)
            supervised_loss = supervised_loss + l * w

        total_loss = supervised_loss + self.matching_loss_weight * matching_loss
        self.log('train/matching_loss', matching_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        return total_loss

    # ------------------- VALIDATION CODE BẮT ĐẦU -------------------

    def on_validation_epoch_start(self):
        self.loss_weights = [base * (exp ** (self.current_epoch / self.max_epochs)) for base, exp in zip(self.loss_weight_bases, self.loss_weight_exps)]
        # Tạo bộ nhớ tích lũy cho dự đoán và ground truth của validation
        self.validation_predictions = {}
        self.validation_ground_truths = {}
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        frames, waves, data = batch
        # Bước 1: Tính dự đoán dưới dạng tensor để tính loss
        predictions_tensor = self.predict(frames)
        
        # Tính loss với tensor (không chuyển sang numpy)
        loss = 0.
        for loss_name, crit, weight in zip(self.loss_names, self.losses, self.loss_weights):
            loss_value = crit(predictions_tensor, waves)
            loss += loss_value * weight
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        
        # Bước 2: Chuyển kết quả dự đoán sang numpy để lưu lại cho metric
        predictions = predictions_tensor.detach().cpu().numpy()
        batch_size = frames.shape[0]
        
        # Bước 3: Đảm bảo dictionary cho dataloader_idx đã được khởi tạo
        if dataloader_idx not in self.validation_predictions:
            self.validation_predictions[dataloader_idx] = {}
            self.validation_ground_truths[dataloader_idx] = {}
        
        # Lặp qua từng phần tử trong batch
        for i in range(batch_size):
            metadata = data[i]
            subject = metadata['subject']
            record = metadata['record']
            idx = metadata['idx']
            
            # Đảm bảo key subject tồn tại bên trong dictionary của dataloader_idx
            if subject not in self.validation_predictions[dataloader_idx]:
                self.validation_predictions[dataloader_idx][subject] = {}
                self.validation_ground_truths[dataloader_idx][subject] = {}
            # Đảm bảo key record tồn tại bên trong dictionary của subject
            if record not in self.validation_predictions[dataloader_idx][subject]:
                self.validation_predictions[dataloader_idx][subject][record] = {}
                self.validation_ground_truths[dataloader_idx][subject][record] = {}
            
            self.validation_predictions[dataloader_idx][subject][record][idx] = predictions[i]
            # Giả sử dữ liệu ground truth cho validation cũng có key 'waves'
            self.validation_ground_truths[dataloader_idx][subject][record][idx] = data[i]['waves'].detach().cpu().numpy()
        
        return loss


    def on_validation_epoch_end(self):
        for dataloader_id in self.validation_predictions.keys():
            predictions = []
            ground_truths = []
            dataloader_predictions = self.validation_predictions[dataloader_id]
            dataloader_ground_truths = self.validation_ground_truths[dataloader_id]

            for subject in dataloader_predictions.keys():
                pred_subj = dataloader_predictions[subject]
                gt_subj = dataloader_ground_truths[subject]
                for record in pred_subj.keys():
                    pred_rec = pred_subj[record]
                    gt_rec = gt_subj[record]
                    pred_ = []
                    gt_ = []
                    for i, idx in enumerate(sorted(pred_rec.keys())):
                        pred = pred_rec[idx]
                        gt = gt_rec[idx]
                        if i > 0:
                            pred = pred[-self.config['data']['chunk_interval']:]
                            gt = gt[-self.config['data']['chunk_interval']:]
                        pred_.append(pred)
                        gt_.append(gt)
                    pred_ = np.concatenate(pred_, axis=0)
                    gt_ = np.concatenate(gt_, axis=0)
                    predictions.append(pred_)
                    ground_truths.append(gt_)

            metrics = calculate_hr_and_hrv_metrics(
                predictions, 
                ground_truths, 
                diff='diff' in self.config['data']['wave_type'][0]
            )
            for metric_name, metric_value in metrics.items():
                self.log(f'val/{dataloader_id}/{metric_name}', metric_value, prog_bar='bpm' in metric_name)
        
        # (2) Ghi toàn bộ metric đã được log vào file
        log_file ="result.log"

        with open(log_file, "a") as f:  # "a" để ghi tiếp vào file mà không ghi đè
            f.write(f"\n\n=== Epoch {self.current_epoch} Validation Metric ===\n")
            for key, value in self.trainer.callback_metrics.items():
                # Lọc những metric liên quan đến validation (nếu muốn)
                if key.startswith("val/"):
                    f.write(f"{key}: {value}\n")
                
        self.validation_predictions = {}
        self.validation_ground_truths = {}
        gc.collect()
        return super().on_validation_epoch_end()

    # ------------------- VALIDATION CODE KẾT THÚC -------------------

    def on_test_epoch_start(self):
        self.predictions = {}
        self.ground_truths = {}
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        frames, waves, data = batch
        predictions = self.predict(frames).squeeze(-1).detach().cpu().numpy()
        batch_size = frames.shape[0]
        for i in range(batch_size):
            metadata = data[i]
            subject = metadata['subject']
            record = metadata['record']
            idx = metadata['idx']
            if dataloader_idx not in self.predictions:
                self.predictions[dataloader_idx] = {}
                self.ground_truths[dataloader_idx] = {}
            if subject not in self.predictions[dataloader_idx]:
                self.predictions[dataloader_idx][subject] = {}
                self.ground_truths[dataloader_idx][subject] = {}
            if record not in self.predictions[dataloader_idx][subject]:
                self.predictions[dataloader_idx][subject][record] = {}
                self.ground_truths[dataloader_idx][subject][record] = {}
            self.predictions[dataloader_idx][subject][record][idx] = predictions[i]
            self.ground_truths[dataloader_idx][subject][record][idx] = data[i]['waves'].detach().cpu().numpy()
        return

    def on_test_epoch_end(self):
        for dataloader_id in self.predictions.keys():
            predictions = []
            ground_truths = []
            dataloader_predictions = self.predictions[dataloader_id]
            dataloader_ground_truths = self.ground_truths[dataloader_id]
            for subject in dataloader_predictions.keys():
                pred_subj = dataloader_predictions[subject]
                gt_subj = dataloader_ground_truths[subject]
                for record in pred_subj.keys():
                    pred_rec = pred_subj[record]
                    gt_rec = gt_subj[record]
                    pred_ = []
                    gt_ = []
                    for i, idx in enumerate(sorted(pred_rec.keys())):
                        pred = pred_rec[idx]
                        gt = gt_rec[idx]
                        if i > 0:
                            pred = pred[-self.config['data']['chunk_interval']:]
                            gt = gt[-self.config['data']['chunk_interval']:]
                        pred_.append(pred)
                        gt_.append(gt)
                    pred_ = np.concatenate(pred_, axis=0)
                    gt_ = np.concatenate(gt_, axis=0)
                    predictions.append(pred_)
                    ground_truths.append(gt_)
            metrics = calculate_hr_and_hrv_metrics(predictions, ground_truths, diff='diff' in self.config['data']['wave_type'][0])
            for metric_name, metric_value in metrics.items():
                self.log(f'test/{dataloader_id}/{metric_name}', metric_value, prog_bar='bpm' in metric_name)
        self.predictions = {}
        self.ground_truths = {}
        gc.collect()
        return super().on_test_epoch_end()

    def val_dataloader(self):
        val_loaders = []
        for args in self.config['data']['val_sets']:
            dataset_cls = get_dataset_cls(args['name'])
            val_set = dataset_cls(**self.config['data']['datasets'][args['name']], split=args['split'], split_idx=self.config.get('split_idx', 0), training=False)
            val_loader = DataLoader(
                val_set,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=False,
                persistent_workers=False,
                collate_fn=collate_fn,
            )
            val_loaders.append(val_loader)
        return val_loaders

    def test_dataloader(self):
        test_loaders = []
        for args in self.config['data']['test_sets']:
            dataset_cls = get_dataset_cls(args['name'])
            test_set = dataset_cls(**self.config['data']['datasets'][args['name']], split=args['split'], training=False)
            test_loader = DataLoader(
                test_set,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=False,
                persistent_workers=False,
                collate_fn=collate_fn,
            )
            test_loaders.append(test_loader)
        return test_loaders

    def configure_optimizers(self):
        optimizer = get_optimizer_cls(self.config['optimizer']['name'])(get_wd_params(self), **self.config['optimizer']['hparams'])
        if 'scheduler' in self.config['optimizer']:
            if self.config['optimizer']['scheduler']['name'] == 'step':
                scheduler = StepLR(optimizer, **self.config['optimizer']['scheduler']['hparams'])
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                    },
                }
            elif self.config['optimizer']['scheduler']['name'] == 'onecycle':
                scheduler = OneCycleLR(optimizer, max_lr=self.config['optimizer']['hparams']['lr'], total_steps=self.num_steps, **self.config['optimizer']['scheduler']['hparams'])
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                    },
                }
        return optimizer

    @property
    def num_steps(self):
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * self.trainer.max_epochs // (self.trainer.accumulate_grad_batches * num_devices)
        return num_steps