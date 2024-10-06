import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor, load, log
from torchmetrics import Accuracy
from train_video import HMDB51DataModule
from typing import Tuple
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
import torchvision.models as models
import datetime

from TADP.tadv_heads import ConvHead

# Define the PyTorch Lightning Module
class ConvClassifier(pl.LightningModule):
    def __init__(self, num_classes, class_names, learning_rate=1e-3):
        super().__init__()

        self.class_names = class_names
        self.model = ConvHead(num_classes=51, in_channels=3, in_dim=512, num_frames=8)
        self.lr = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)

    def _print_log(self, logits, labels):
        probs = F.softmax(logits, 1).detach().cpu().numpy()
        preds = torch.argmax(logits, 1).detach().cpu().numpy()
        pred_labels = [pred for pred in preds] if len(self.class_names) < 51 else [self.class_names[pred] for pred in preds]
        print('\npredictions:', pred_labels)
        print('probabilities of preds:', [probs[i][c] for i,c in enumerate(preds)])
        print('truth:', [self.class_names[truth] for truth in labels.detach().cpu().numpy()], '\n')

    def forward(self, x):
        if type(x) is list:
            x = torch.stack(x)
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        frames, labels, metas = batch
        logits = self(frames)
        # self._print_log(logits, labels)
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(logits, labels)
        self.log_dict({
            'train_loss': loss, 
            'train_acc': accuracy
        }, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, labels, metas = batch
        logits = self(frames)
        if batch_idx % 10 == 0:
            self._print_log(logits, labels)
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(logits, labels)
        self.log_dict({
            'val_loss': loss, 
            'val_acc': accuracy
        }, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        frames, labels, metas = batch
        logits = self(frames)
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(logits, labels)
        self.log_dict({
            'test_loss': loss, 
            'test_acc': accuracy
        }, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default='hmdb_resnet')
    parser.add_argument("--cls_head", type=str, default='transformer')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=4)
    parser.add_argument("--log_model_every_n_epochs", type=int, default=-1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--wandb_group", type=str, default="mcrco")

    # debugging presets
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--no_wandb", action='store_true', default=False)
    parser.add_argument("--val_debug", action='store_true', default=False)
    parser.add_argument("--wandb_debug", action='store_true', default=False)
    # test remote machine if it is working without wasting time downloading datasets
    parser.add_argument("--test_machine", action='store_true', default=False)

    # experiment parameters
    parser.add_argument("--pretrained", type=int, default=True)
    parser.add_argument("--frozen", type=int, default=True)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument('--optimizer_config_preset', type=int, default=0)
    parser.add_argument('--strategy', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument("--accum_grad_batches", type=int, default=1)
    parser.add_argument('--trainer_ckpt_path', type=str, default=None)
    parser.add_argument('--save_checkpoint_path', type=str, default='out/')
    parser.add_argument('--train_debug', action='store_true', default=False)
    args = parser.parse_args()

    pretrained = args.pretrained
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    log_freq = args.log_freq
    log_every_n_steps = args.log_every_n_steps
    wandb_group = args.wandb_group
    wandb_name = args.exp_name
    strategy = args.strategy
    accum_grad_batches = args.accum_grad_batches
    save_checkpoint_path = args.save_checkpoint_path

    save_topk = 1
    save_last = True
    limit_train_batches = None
    limit_val_batches = None
    if args.debug:
        max_epochs = 4
        os.environ["WANDB_MODE"] = "dryrun"
        num_workers = 0
        batch_size = batch_size
        log_freq = 1
        save_last = False
        save_topk = 0
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    if args.wandb_debug:
        num_workers = 0
        batch_size = 16 if 'TADP' not in args.model else batch_size
        limit_val_batches = 2
        limit_train_batches = 2
        wandb_group = "wandb_debugging"
        wandb_name = f"dummy_{datetime.datetime.now().__str__()}"
        save_last = False
        save_topk = 0
    if args.val_debug:
        limit_val_batches = 2
        limit_train_batches = 2
        os.environ["WANDB_MODE"] = "dryrun"
    if args.test_machine:
        args.train_dataset = ['dummy_data']
        args.eval_dataset = ['dummy_data']
        os.environ["WANDB_MODE"] = "dryrun"

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')

    # set up datamodule
    base_path = './data/'
    dataset_name = 'hmdb51'
    max_vid_frames = 60 # already evenly sampled 60 frames from hmdb
    if args.debug:
        dataset_name = 'hmdb-small'
    datamodule = HMDB51DataModule(
        os.path.join(base_path, f'{dataset_name}/sampled_videos'),
        os.path.join(base_path, f'{dataset_name}/split_files'),
        num_frames=8,
        max_frames = max_vid_frames,
        step=max_vid_frames + 10, # + 10 to be safe it's 1 clip per vid
        format="TCHW",
        batch_size=batch_size,
        num_workers=num_workers
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'./checkpoints/{args.exp_name}/',
        filename='model_checkpoint_{epoch}',
        # save_top_k=save_topk,  # Save top1 Why?? this is 40GB of checkpoints -->> # Save all checkpoints.
        save_top_k=save_topk,
        save_last=save_last,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_callback, checkpoint_callback]

    logger = pl.loggers.WandbLogger(
        name=wandb_name,
        group=wandb_group or "mcrco",
        project="tadvar",
        log_model=True,
    )

    model = ConvClassifier(num_classes=51, class_names=datamodule.train.classes, learning_rate=1e-3)

    # watch model
    logger.watch(model, log="all", log_freq=log_freq)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=strategy if strategy != '' else 'auto',  # check somehow ddp is using more gpu memory than auto
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        limit_train_batches=limit_train_batches,  # None unless --wandb_debug or --val_debug flag is set
        limit_val_batches=limit_val_batches,  # None unless --wandb_debug or --val_debug flag is set
        check_val_every_n_epoch=args.check_val_every_n_epoch,  # None unless --wandb_debug flag is set
        sync_batchnorm=True if args.num_gpus > 1 else False,
        accumulate_grad_batches=accum_grad_batches,
        gradient_clip_val=12.0
    )
    if trainer.global_rank == 0:
        logger.experiment.config.update(args)

    trainer.fit(model, datamodule, ckpt_path=args.trainer_ckpt_path)

    if not args.debug:
        save_model_name = f'{args.exp_name}.ckpt'
        if save_checkpoint_path != '':
            if not os.path.exists(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)
            trainer.save_checkpoint(os.path.join(save_checkpoint_path, save_model_name))
        trainer.test(model, datamodule, ckpt_path=os.path.join(save_checkpoint_path, save_model_name))

if __name__ == "__main__":
    main()

