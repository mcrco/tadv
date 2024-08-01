import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchvision.datasets import HMDB51
from torchvision.ops.boxes import torchvision
from typing import Tuple
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
import torchvision.models as models

import datetime

class HMDB51WithMetadata(HMDB51):
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int, str]:
        video, audio, _, video_idx = self.video_clips.get_clip(idx)
        video_path = self.video_clips.metadata['video_paths'][video_idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, class_index, video_name

class HMDB51DataModule(pl.LightningDataModule):
    def __init__(self, video_path, split_file_path, num_frames, max_frames, step, format, batch_size, num_workers, normalize_videos):
        super().__init__()
        self.video_path = video_path
        self.split_file_path = split_file_path
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.step = step
        self.format = format
        self.batch_size = batch_size
        self.num_workers = num_workers

        def convert_to_float(x):
            if type(x) is list:
                x = torch.stack(x)
            if isinstance(x, torch.Tensor) and x.type() != torch.cuda.FloatTensor:
                x = x.float()
            return x

        frame_resize = torchvision.transforms.Resize((512, 512))
        def resize_transform(video):
            trans_frames = []
            for frame in video:
                trans_frames.append(frame_resize(frame))
            return torch.stack(trans_frames)

        norm_fn = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        def normalize(video):
            norm_frames = [norm_fn(frame) for frame in video]
            return torch.stack(norm_frames)

        sample_frames = T2.UniformTemporalSubsample(self.num_frames)

        def transform(video):
            video = sample_frames(video)
            video = convert_to_float(video)
            if normalize_videos: 
                video = normalize(video)
            return resize_transform(video)

        self.train = HMDB51WithMetadata(self.video_path, 
            self.split_file_path,
            frames_per_clip=self.max_frames,
            step_between_clips=self.step,
            output_format=self.format,
            transform=transform,
            train=True
        )
        self.test = HMDB51WithMetadata(self.video_path, 
            self.split_file_path,
            frames_per_clip=self.max_frames,
            step_between_clips=self.step,
            output_format=self.format,
            transform=transform,
            train=False
        )

        def collate_function(data):
            videos = [vid[0].to(dtype=torch.float32) for vid in data]
            video_names = [vid[3] for vid in data]
            labels = torch.tensor([vid[2] for vid in data], dtype=torch.long)
            return videos, labels, video_names
        self.collate_function = collate_function

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_function)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_function)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_function)

# Define the PyTorch Lightning Module
class ResNet50VideoClassifier(pl.LightningModule):
    def __init__(self, num_classes, num_frames, learning_rate=1e-3, pretrained=True):
        super().__init__()

        # Load the pretrained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=pretrained, progress=True)
        
        # Replace the final fully connected layer with an identity layer
        self.resnet50.fc = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1000 * num_frames, 512),
            nn.SiLU(),
            nn.Linear(512, num_classes)
        )

        self.lr = learning_rate

        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)

    def forward(self, x):
        # Extract features using ResNet-50
        print(x.shape)
        n, f, c, h, w = x.shape
        x = x.view(n * f, c, h, w)
        features = self.resnet50(x)
        print(features.shape)
        features = features.view(n, f, features.shape[1:])
        print(features.shape)
        # Classify using the classification head
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        frames, labels, metas = batch
        logits = self(frames)
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
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(logits, labels)
        self.log_dict({
            'train_loss': loss, 
            'train_acc': accuracy
        }, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        frames, labels, metas = batch
        logits = self(frames)
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(logits, labels)
        self.log_dict({
            'train_loss': loss, 
            'train_acc': accuracy
        }, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--log_model_every_n_epochs", type=int, default=-1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--wandb_group", type=str, default="mcrco")

    # debugging presets
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--val_debug", action='store_true', default=False)
    parser.add_argument("--wandb_debug", action='store_true', default=False)
    # test remote machine if it is working without wasting time downloading datasets
    parser.add_argument("--test_machine", action='store_true', default=False)

    # experiment parameters
    parser.add_argument("--from_scratch", action='store_true', default=False)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument('--optimizer_config_preset', type=int, default=0)
    parser.add_argument('--strategy', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument("--accum_grad_batches", type=int, default=1)
    parser.add_argument('--normalize_videos', action='store_true', default=True)
    parser.add_argument('--trainer_ckpt_path', type=str, default=None)
    parser.add_argument('--save_checkpoint_path', type=str, default='out/')
    parser.add_argument('--train_debug', action='store_true', default=False)
    args = parser.parse_args()

    pretrained = not args.from_scratch
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
        os.path.join(base_path, f'{dataset_name}/videos'), 
        os.path.join(base_path, f'{dataset_name}/split_files'),
        num_frames=8,
        max_frames = max_vid_frames,
        step=max_vid_frames + 10, # + 10 to be safe it's 1 clip per vid
        format="TCHW",
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_videos=args.normalize_videos
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
        name=wandb_name or "segmentation_test, model={}".format(model_name) + "usingDecoderFeatures={}".format(
            use_decoder_features),
        group=wandb_group or "mcrco",
        project="tadvar",
        log_model=True,
    )

    model = ResNet50VideoClassifier(51, learning_rate=1e-3, pretrained=pretrained)

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
        accumulate_grad_batches=accum_grad_batches
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

