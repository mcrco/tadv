import json
import os
import pickle
import warnings
from typing import Dict, List, Tuple

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torch import nn
from torchmetrics import Accuracy

import wandb
from TADV.t2v_utils.svd_extract import pipe_features
from TADV.tadv_heads import LinearHead, MLPHead, NeeharHead, RogerioHead


class TADPVid(pl.LightningModule):

    def __init__(
        self,
        video_model="ali-vilab/text-to-video-ms-1.7b",
        text_conditioning="blip",
        blip_caption_path=None,
        cls_head="mlp",
        use_decode_head=True,
        class_names=None,
        freeze_backbone=False,
        log_ca=True,
        max_epochs=50,
        dataset_len=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.video_pipe = DiffusionPipeline.from_pretrained(
            video_model, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        self.freeze_backbone = freeze_backbone
        self.log_ca = log_ca

        if class_names is not None:
            self.n_classes = len(class_names)
        self.class_names = class_names
        self.max_epochs = max_epochs
        self.dataset_len = dataset_len
        self.metric = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=51, top_k=1)

        self.text_conditioning = text_conditioning
        self.blip_caption_path = blip_caption_path
        # turn text conditioning into list
        if "+" in self.text_conditioning:
            self.text_conditioning = self.text_conditioning.split("+")
        else:
            self.text_conditioning = [self.text_conditioning]

        if "blip" in self.text_conditioning:
            with open(self.blip_caption_path, "r") as f:
                self.blip_captions = json.load(f)

        self.cls_head = cls_head
        self.use_decode_head = use_decode_head
        if use_decode_head:
            if self.cls_head == "neehar":
                self.decode_head = NeeharHead(num_classes=51)
            if self.cls_head == "linear":
                self.decode_head = LinearHead(num_classes=51, in_channels=320 * 64 * 64)
            if self.cls_head == "rogerio":
                self.decode_head = RogerioHead(num_classes=51, in_channels=717)
            if self.cls_head == "mlp":
                self.decode_head = MLPHead(num_classes=51, in_channels=397)
        self.batchnorm = nn.BatchNorm2d(397)

        ## test for now fix the unet
        if self.freeze_backbone:
            for param in self.video_pipe.parameters():
                param.requires_grad = False

    def initialize_model(self):
        pass

    def extract_feat(self, video, caption):
        # get latents
        video = video.half()
        with torch.no_grad():
            latents = self.video_pipe.vae.encode(video).latent_dist.sample()
        latents = latents.permute(1, 0, 2, 3).unsqueeze(0)

        features, cross_attns, _ = pipe_features(
            pipe=self.video_pipe, prompt=caption, latents=latents
        )
        return torch.cat((features[0], cross_attns[0]), 1)

    def forward(self, x, img_metas=None):
        if type(x) == list:
            x = torch.stack(x)

        captions = None
        if self.blip_captions is not None and img_metas is not None:
            captions = [self.blip_captions[name] for name in img_metas]

        n_batch = x.shape[0]
        all_features = []
        if captions is None:
            captions = [None] * n_batch
        for vid, caption in zip(x, captions):
            all_features.append(self.extract_feat(vid, caption))
        features = torch.cat(all_features)
        n_channels = features.shape[0] // n_batch
        features = features.float()
        features = self.batchnorm(features)
        features = features.reshape(n_batch, n_channels, *features.shape[1:])

        if not self.use_decode_head:
            return features

        x = self.decode_head(features)
        return x

    def configure_optimizers(self):
        # TODO: double check here
        optimizer = torch.optim.Adam(self.decode_head.parameters(), lr=0.001)
        # have differernt learning rate for different layers
        # parameters to optimize
        # lesslr_no_decay = list()
        # lesslr_decay = list()
        # no_lr = list()
        # no_decay = list()
        # decay = list()
        # for name, m in self.named_parameters():
        #     if 'unet' in name and 'norm' in name:
        #         lesslr_no_decay.append(m)
        #     elif 'unet' in name:
        #         lesslr_decay.append(m)
        #     elif 'encoder_vq' in name:
        #         no_lr.append(m)
        #     elif 'norm' in name:
        #         no_decay.append(m)
        #     else:
        #         decay.append(m)

        # if self.freeze_backbone:
        #     params_to_optimize = [
        #         {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.0},
        #         {'params': lesslr_decay, 'lr_scale': 0.0},
        #         {'params': no_lr, 'lr_scale': 0.0},
        #         {'params': no_decay, 'weight_decay': 0.0},
        #     ]
        # else:
        #     params_to_optimize = [
        #         {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.01},
        #         {'params': lesslr_decay, 'lr_scale': 0.01},
        #         {'params': no_lr, 'lr_scale': 0.0},
        #         {'params': no_decay, 'weight_decay': 0.0},
        #         {'params': decay}
        #     ]
        # optimizer = torch.optim.AdamW(params_to_optimize,
        #                               lr=1e-4,
        #                               weight_decay=1e-2,
        #                               amsgrad=False
        #                              )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda x: (1 - x / (self.dataset_len * self.max_epochs)) ** 0.9,
        )

        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}
        ]

    def _print_outputs(self, outputs, y):
        probs = F.softmax(outputs, 1).detach().cpu().numpy()
        preds = torch.argmax(outputs, 1).detach().cpu().numpy()
        print("predictions:", [self.class_names[pred] for pred in preds])
        print("probabilities of preds:", [probs[i][c] for i, c in enumerate(preds)])
        print("truth:", [self.class_names[truth] for truth in y.detach().cpu().numpy()])
        if self.decode_head.conv_down[0].weight.grad is not None:
            print("conv 0 grad", self.decode_head.conv_down[0].weight.grad.abs().max())

    def training_step(self, batch, batch_idx):
        x, y, metas = batch
        outputs = self(x, img_metas=metas)

        loss = self.metric(outputs, y)
        accuracy = self.accuracy(outputs, y)
        self.log_dict({"train_loss": loss, "train_acc": accuracy}, sync_dist=True)
        return loss

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, strict=True):
        # TODO diff be
        self.load_state_dict(torch.load(path)["state_dict"], strict=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, metas = batch
        preds = self(x, img_metas=metas)
        loss = self.metric(preds, y)
        accuracy = self.accuracy(preds, y)
        self.log_dict({"test_loss": loss, "test_acc": accuracy}, sync_dist=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, metas = batch
        outputs = self(x, img_metas=metas)

        self._print_outputs(outputs, y)

        loss = self.metric(outputs, y)
        accuracy = self.accuracy(outputs, y)
        self.log_dict({"val_loss": loss, "val_acc": accuracy}, sync_dist=True)

        # log video and attention maps every 10 batches in each validation loop
        if batch_idx % 10 == 0:
            video, label = x[0], y[0]
            # if type(video) is list:
            #     video = torch.stack(video)

            # log diffusion cross attention maps from middle (4th) frame of each video
            caption = None
            if self.blip_captions is not None:
                caption = self.blip_captions[metas[0]]
            features = self.extract_feat(video, caption)
            ca_maps = features[3].cpu().numpy()[320:335]  # 4th frame of video
            np.reshape(ca_maps, (*ca_maps.shape, 1))
            print(ca_maps.shape)

            def unnormalize(img_tensor, mean=[128, 128, 128], std=[128, 128, 128]):
                mean = torch.tensor(mean).view(1, 3, 1, 1)
                std = torch.tensor(std).view(1, 3, 1, 1)
                unnormalized_img = img_tensor * std + mean
                return unnormalized_img

            frames = unnormalize(video).cpu().numpy().astype(int)

            if self.log_ca:
                self.logger.experiment.log(
                    {
                        "video": wandb.Video(frames, caption=self.class_names[label]),
                        "cross_att_maps": [
                            wandb.Image(ca_map, caption=self.blip_captions[metas[0]])
                            for ca_map in ca_maps
                        ],
                    }
                )

    def _log_wandb_media(self, video, label, meta):
        # log diffusion cross attention maps from middle (4th) frame of each video
        caption = None
        if self.blip_captions is not None:
            caption = self.blip_captions[meta]
        features = self.extract_feat(video, caption)
        ca_maps = features[:][320:]
        ca_maps = torch.stack(
            [
                T.Normalize(mean=(0.5,), std=(0.25,))(ca_map.unsqueeze(0)).permute(
                    1, 2, 0
                )
                for ca_map in ca_maps
            ]
        )
        ca_maps = ca_maps.detach().cpu().numpy()

        tokens = ["" for _ in range(16)]
        if caption is not None:
            _, t = self.sd_model.get_learned_conditioning(caption, return_tokens=True)
            tokens = [
                self.sd_model.cond_stage_model.tokenizer.decode(ind) for ind in t[0]
            ]

        def unnormalize(
            img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ):
            mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
            std = torch.tensor(std).view(1, 3, 1, 1).cuda()
            unnormalized_img = img_tensor * std + mean
            return unnormalized_img

        frames = unnormalize(video).cpu().numpy()

        if self.log_ca:
            self.logger.experiment.log(
                {
                    "video": wandb.Video(frames, caption=self.class_names[label]),
                    "cross_att_maps": [
                        wandb.Image(ca_map, caption=token)
                        for ca_map, token in zip(ca_maps, tokens)
                    ],
                }
            )


def tensor_to_video(tensor, output_file="out/output_video.avi", fps=30):
    frames = tensor.cpu().numpy()  # Convert PyTorch tensor to numpy array

    # Transpose to (num_frames, height, width, channels) for OpenCV compatibility
    frames = frames.transpose((0, 2, 3, 1))

    # OpenCV video writer
    _, height, width, _ = frames.shape  # shape is (num_frames, height, width, channels)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame in frames:
        frame = np.uint8(frame)  # Convert frame to uint8
        out.write(frame)

    out.release()
