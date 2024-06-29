import json
import os
import pickle
import warnings
from typing import Dict, List, Tuple

import lightning.pytorch as pl
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import torchvision
import torchvision.transforms.v2.functional
import wandb
from PIL import Image
from chainercv.evaluations import eval_detection_voc
# our stuff
from diffusers.utils.hub_utils import _get_model_file
from huggingface_hub.constants import HF_HUB_CACHE as DIFFUSERS_CACHE
### tadp crossdomain object detection
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch import nn, Tensor
from torchvision import datapoints
from torchvision.models.detection.anchor_utils import AnchorGenerator
from detectron2.structures.instances import Instances
from ldm.util import instantiate_from_config

from TADP.vpd.models import UNetWrapper, TextAdapter

hacky = False

class BasicDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = lambda x : torch.mean(x, 1) # first dim is video, second dim is frames
        self.flatten = nn.Flatten()
        self.cls_head = nn.Linear(320 * 64 * 64, 51) # 51 classes in HMDB

    def forward(self, x):
        x = self.mean(x)
        x = self.flatten(x)
        x = self.cls_head(x)
        return x

class TADPVid(pl.LightningModule):

    def __init__(self,
                 sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
                 unet_config=dict(),
                 class_embedding_path='ade_class_embeddings.pth',
                 gamma_init_value=1e-4,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 use_decoder=False,
                 cfg=None,
                 class_names=None,
                 freeze_backbone=False,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        # get config from *args and **kwargs

        self.freeze_backbone = freeze_backbone

        if class_names is not None:
            self.n_classes = len(class_names)
        self.metric = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=51, top_k=1)
        self.dataset_name = "voc"

        self.cfg = cfg
        try:
            self.object_dataloader_indices = kwargs['object_dataloader_indices']
        except KeyError:
            self.object_dataloader_indices = None

        # turn text conditioning into list
        if '+' in self.cfg['text_conditioning']:
            self.cfg['text_conditioning'] = self.cfg['text_conditioning'].split('+')
        else:
            self.cfg['text_conditioning'] = [self.cfg['text_conditioning']]

        ### check if model is there if not DL
        self.text2imgModel = None
        ckpt = "v1-5-pruned-emaonly.ckpt"
        repo = "runwayml/stable-diffusion-v1-5"
        out_dir = "checkpoints"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(os.path.join(out_dir, ckpt)):
            hf_hub_download(repo_id=repo, filename=ckpt, local_dir=out_dir)

        config = OmegaConf.load('stable_diffusion/configs/stable-diffusion/v1-inference.yaml')
        config.model.params.ckpt_path = f'./{sd_path}'
        # if ('blip' not in self.cfg["text_conditioning"] and
        #         'class_names' not in self.cfg['text_conditioning'] and
        #         'textual_inversion' not in self.cfg['text_conditioning']):
        #     config.model.params.cond_stage_config.target = 'stable_diffusion.ldm.modules.encoders.modules.AbstractEncoder'

        sd_model = instantiate_from_config(config.model)

        if self.cfg['dreambooth_checkpoint'] is not None:
            sd_model.load_state_dict(torch.load(self.cfg['dreambooth_checkpoint'])['state_dict'], strict=False)
            print('Loaded dreambooth checkpoint!')

        # handle logic for using scaled encoder
        if not self.cfg.get('use_scaled_encode', False):
            self.encoder_vq = sd_model.first_stage_model
            sd_model.first_stage_model = None
            if not use_decoder:
                del self.encoder_vq.decoder
            ### set grads to zero to be safe
            for param in self.encoder_vq.parameters():
                param.requires_grad = False
        else:
            if not use_decoder:
                del sd_model.first_stage_model.decoder

        self.model = UNetWrapper(sd_model.model, **unet_config)
        sd_model.model = None
        keep_cond = False
        if 'blip' in self.cfg["text_conditioning"]:
            with open(self.cfg['blip_caption_path'], 'r') as f:
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])
            for param in sd_model.cond_stage_model.parameters():
                param.requires_grad = False
            keep_cond = True
        # if 'textual_inversion' in self.cfg["text_conditioning"]:
        #     with open(self.cfg['textual_inversion_caption_path'], 'r') as f:
        #         print('Loaded textual inversion captions!')
        #         self.textual_inversion_captions = json.load(f)
        #         # get max length
        #         self.textual_inversion_max_length = max([len(caption) for caption in self.textual_inversion_captions])
        #     for param in sd_model.cond_stage_model.parameters():
        #         param.requires_grad = False
        #     keep_cond = True
        if 'blip' in self.cfg['text_conditioning'] and self.cfg['cross_blip_caption_path'] is not None:
            with open(self.cfg['cross_blip_caption_path'], 'r') as f:
                self.cross_blip_captions = json.load(f)
                # get max length
                self.cross_blip_max_length = max([len(caption) for caption in self.cross_blip_captions])
        else:
            self.cross_blip_captions = None

        if 'class_names' in self.cfg['text_conditioning']:
            self.class_names = self.cfg['class_names']
            with torch.no_grad():
                sd_model.cond_stage_model.to('cuda')
                class_emb_stack = []
                for class_name in self.class_names:
                    emb = sd_model.get_learned_conditioning(class_name)[[0], 1]
                    class_emb_stack.append(emb)
                self.class_names_embs = torch.stack(class_emb_stack, dim=1)

        # if not keep_cond:
        #     del sd_model.cond_stage_model

        self.use_decoder = use_decoder
        self.sd_model = sd_model

        # class embeddings & text adapter
        # TODO: implement me

        # check if class_embedding_path exists
        if not os.path.exists(class_embedding_path):
            print('No class embeddings provided!, please run create_class_embeddings.py --dataset pascal')

        # check if using the correct class embeddings
        # assert class_embeddings.size(0) == self.n_classes

        self.with_neck = True

        self.decode_head = BasicDecoder()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # assert self.with_decode_head

        ## test for now fix the unet
        for param in self.model.parameters():
            param.requires_grad = False

        textual_inversion_token_path = self.cfg['textual_inversion_token_path']
        if textual_inversion_token_path is not None:
            self.text_encoder = sd_model.cond_stage_model.transformer
            self.tokenizer = sd_model.cond_stage_model.tokenizer
            self._load_textual_inversion_token(textual_inversion_token_path)

    def _load_textual_inversion_token(self, token):
        token_ids_and_embeddings = []

        # 1. Load textual inversion file
        model_file = _get_model_file(
            token,
            weights_name="learned_embeds.bin",
            cache_dir=DIFFUSERS_CACHE,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=False,
            use_auth_token=None,
            revision=None,
            subfolder=None,
            user_agent={"file_type": "text_inversion", "framework": "pytorch"},
        )
        state_dict = torch.load(model_file, map_location="cpu")

        # Save text_encoder and tokenizer

        # 2. Load token and embedding correctly from file
        loaded_token, embedding = next(iter(state_dict.items()))
        token = loaded_token
        embedding = embedding.to(dtype=self.text_encoder.dtype, device=self.text_encoder.device)

        # 3. Make sure we don't mess up the tokenizer or text encoder
        vocab = self.tokenizer.get_vocab()
        if token in vocab:
            raise ValueError(
                f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
            )

        tokens = [token]
        embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

        # add tokens and get ids
        self.tokenizer.add_tokens(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids_and_embeddings += zip(token_ids, embeddings)

        print(f"Loaded textual inversion embedding for {token}.")

        # resize token embeddings and set all new embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        for token_id, embedding in token_ids_and_embeddings:
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding

    def initialize_model(self):
        pass

    def create_text_embeddings(self, latents, img_metas=None, captions=None, texts=None):
        texts = []
        _cs = []
        conds = []
        if captions is None:
            for l in latents:
                text = 'human'
                c = self.sd_model.get_learned_conditioning(text)
                texts.append(text)
                _cs.append(c)
        else:
            for caption in captions:
                text = caption
                c = self.sd_model.get_learned_conditioning(text)
                texts.append(text)
                _cs.append(c)
        c = torch.cat(_cs, dim=0)
        conds.append(c)
        c_crossattn = torch.cat(conds, dim=1)
        return c_crossattn

    def extract_feat(self, img, img_metas=None, captions=None, texts=None):
        """Extract features from images."""
        if self.cfg.get('use_scaled_encode', False):
            with torch.no_grad():
                latents = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(img))
        else:
            with torch.no_grad():
                latents = self.encoder_vq.encode(img)
            latents = latents.mode().detach()

        viz = False

        c_crossattn = self.create_text_embeddings(latents, img_metas=img_metas, captions=captions, texts=texts)

        t = torch.from_numpy(np.array([1])).to(img.device)

        outs = self.model(latents, t, c_crossattn=[c_crossattn])

        if viz:
            # vis
            plt.imshow(outs[1][0][1].detach().cpu().numpy())
            plt.show()
            #
            plt.imshow(outs[1][0][-2].detach().cpu().numpy())
            plt.show()

        return outs

    def forward(self, x, img_metas=None):
        captions = None
        if self.blip_captions is not None and img_metas is not None:
            captions = [self.blip_captions[name] for name in img_metas]
        video_features = []
        for i in range(len(x)):
            video = x[i]
            frame_captions = [captions[i] for _ in range(video.shape[0])] if captions is not None else None
            features = self.extract_feat(video, captions=frame_captions)
            video_features.append(features[0]) 
        features = torch.stack(video_features)
        x = self.decode_head(features) 

        return x

    def inference(self, img, captions=None):
        x = [torchvision.transforms.ToTensor()(x) for x in img]

        orig_images = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])(x) for x in x]
        orig_images = [datapoints.Image(x) for x in orig_images]

        from torchvision.transforms import v2 as T
        _size = 512
        trans = T.Compose(
            [
                T.Resize((_size, _size)),
            ]
        )
        orig_images = trans(orig_images)

        orig_images = torch.stack(orig_images)
        orig_images = torchvision.models.detection.image_list.ImageList(orig_images,
                                                                        image_sizes=[(_size, _size)] * len(orig_images))

        orig_images_tensors = orig_images.tensors.to(self.device)

        features = self.extract_feat(orig_images_tensors, texts=captions)

        x = self.decode_head(features[0])

        return x

    def configure_optimizers(self):
        # TODO: double check here
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # have differernt learning rate for different layers
        # parameters to optimize
        lesslr_no_decay = list()
        lesslr_decay = list()
        no_lr = list()
        no_decay = list()
        decay = list()
        for name, m in self.named_parameters():
            if 'unet' in name and 'norm' in name:
                lesslr_no_decay.append(m)
            elif 'unet' in name:
                lesslr_decay.append(m)
            elif 'encoder_vq' in name:
                no_lr.append(m)
            elif 'norm' in name:
                no_decay.append(m)
            else:
                decay.append(m)

        if self.freeze_backbone:
            params_to_optimize = [
                {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.0},
                {'params': lesslr_decay, 'lr_scale': 0.0},
                {'params': no_lr, 'lr_scale': 0.0},
                {'params': no_decay, 'weight_decay': 0.0},
            ]
        else:
            params_to_optimize = [
                {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.01},
                {'params': lesslr_decay, 'lr_scale': 0.01},
                {'params': no_lr, 'lr_scale': 0.0},
                {'params': no_decay, 'weight_decay': 0.0},
                {'params': decay}
            ]
        optimizer = torch.optim.AdamW(params_to_optimize,
                                      lr=0.00001,
                                      # lr=0.000005,
                                      weight_decay=1e-2,
                                      amsgrad=False
                                      )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (1 - x / (
                self.cfg["dataset_len"] * self.cfg["max_epochs"])) ** 0.9)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}]

    def training_step(self, batch, batch_idx):
        x, y, metas = batch
        preds = self(x, img_metas=metas)
        return self.metric(preds, y)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, strict=True):
        # TODO diff be
        self.load_state_dict(torch.load(path)['state_dict'], strict=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, metas = batch
        preds = self(x, img_metas=metas)
        loss = self.metric(preds, y)
        accuracy = self.accuracy(preds, y)
        self.log_dict({
            'val_loss': loss, 
            'val_acc': accuracy
        }, sync_dist=True)

        # log video and attention maps for first batch in each validation step
        if batch_idx == 0:
            # log first video
            video = x[0]
            frames = video.cpu().numpy()
            
            # log diffusion cross attention maps from middle (4th) frame of first video
            caption = None
            if self.blip_captions is not None:
                caption = self.blip_captions[metas[0]]
            frame_captions = [caption for _ in range(video.shape[0])] if caption is not None else None
            features = self.extract_feat(video, captions=frame_captions)
            ca_maps = features[0].cpu().numpy()[3] # 4th frame of first video
            np.reshape(ca_maps, (*ca_maps.shape, 1))

            self.logger.experiment.log({
                'video': wandb.Video(frames, caption=metas[0]),
                'cross_att_maps': [wandb.Image(ca_map, caption=self.blip_captions[metas[0]]) for ca_map in ca_maps]
            })



