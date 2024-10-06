import json
import os

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.v2.functional
# our stuff
from diffusers.utils.hub_utils import _get_model_file
### tadp crossdomain object detection
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE as DIFFUSERS_CACHE
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torchmetrics import Accuracy

import wandb
# from torchvision import datapoints
from ldm.util import instantiate_from_config
from TADP.vpd.models import TextAdapter, UNetWrapper
from TADV.tadv_heads import (ConvHead, ConvMLP, LinearHead, MLPHead,
                             NeeharHead, RogerioHead)


class TADPVid(pl.LightningModule):

    def __init__(
        self,
        sd_path="checkpoints/v1-5-pruned-emaonly.ckpt",
        unet_config=dict(),
        class_embedding_path="ade_class_embeddings.pth",
        train_cfg=None,
        test_cfg=None,
        use_decoder=False,
        use_decode_head=True,
        cfg=None,
        class_names=None,
        freeze_backbone=False,
        log_ca=False,
        head_cfg=dict(),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # get config from *args and **kwargs

        self.freeze_backbone = freeze_backbone
        self.head_cfg = head_cfg
        self.log_ca = log_ca

        if class_names is not None:
            self.n_classes = len(class_names)
        self.class_names = class_names
        self.metric = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(
            task="multiclass", num_classes=self.head_cfg["num_classes"], top_k=1
        )
        self.dataset_name = "voc"

        self.cfg = cfg
        try:
            self.object_dataloader_indices = kwargs["object_dataloader_indices"]
        except KeyError:
            self.object_dataloader_indices = None

        # turn text conditioning into list
        if "+" in self.cfg["text_conditioning"]:
            self.cfg["text_conditioning"] = self.cfg["text_conditioning"].split("+")
        else:
            self.cfg["text_conditioning"] = [self.cfg["text_conditioning"]]

        ### check if model is there if not DL
        self.text2imgModel = None
        ckpt = "v1-5-pruned-emaonly.ckpt"
        repo = "runwayml/stable-diffusion-v1-5"
        out_dir = "checkpoints"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(os.path.join(out_dir, ckpt)):
            hf_hub_download(repo_id=repo, filename=ckpt, local_dir=out_dir)

        config = OmegaConf.load(
            "stable_diffusion/configs/stable-diffusion/v1-inference.yaml"
        )
        config.model.params.ckpt_path = f"./{sd_path}"
        # if ('blip' not in self.cfg["text_conditioning"] and
        #         'class_names' not in self.cfg['text_conditioning'] and
        #         'textual_inversion' not in self.cfg['text_conditioning']):
        #     config.model.params.cond_stage_config.target = 'stable_diffusion.ldm.modules.encoders.modules.AbstractEncoder'

        sd_model = instantiate_from_config(config.model)

        if self.cfg["dreambooth_checkpoint"] is not None:
            sd_model.load_state_dict(
                torch.load(self.cfg["dreambooth_checkpoint"])["state_dict"],
                strict=False,
            )
            print("Loaded dreambooth checkpoint!")

        # handle logic for using scaled encoder
        if not self.cfg.get("use_scaled_encode", False):
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

        self.model = UNetWrapper(sd_model.model, **unet_config, base_size=512)
        sd_model.model = None
        if "blip" in self.cfg["text_conditioning"]:
            with open(self.cfg["blip_caption_path"], "r") as f:
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max(
                    [len(caption) for caption in self.blip_captions]
                )
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
        if (
            "blip" in self.cfg["text_conditioning"]
            and self.cfg["cross_blip_caption_path"] is not None
        ):
            with open(self.cfg["cross_blip_caption_path"], "r") as f:
                self.cross_blip_captions = json.load(f)
                # get max length
                self.cross_blip_max_length = max(
                    [len(caption) for caption in self.cross_blip_captions]
                )
        else:
            self.cross_blip_captions = None

        if "class_names" in self.cfg["text_conditioning"]:
            self.class_names = self.cfg["class_names"]
            with torch.no_grad():
                sd_model.cond_stage_model.to("cuda")
                class_emb_stack = []
                for class_name in self.class_names:
                    # print(class_name)
                    emb = sd_model.get_learned_conditioning(class_name)[[0], 1]
                    class_emb_stack.append(emb)
                self.class_names_embs = torch.stack(class_emb_stack, dim=1)

        self.use_decoder = use_decoder
        self.sd_model = sd_model

        self.use_decode_head = use_decode_head
        if use_decode_head:
            if self.cfg["cls_head"] == "neehar":
                self.decode_head = NeeharHead(
                    num_classes=self.head_cfg['num_classes'],
                    in_channels=self.head_cfg['num_channels'],
                    hidden_dim=self.head_cfg['hidden_dim'],
                    dropout=self.head_cfg['dropout'],
                    num_frames=self.head_cfg['num_frames'],
                )
            elif self.cfg["cls_head"] == "linear":
                self.decode_head = LinearHead(
                    num_classes=51, in_channels=self.head_cfg["num_channels"] * 64 * 64
                )
            elif self.cfg["cls_head"] == "rogerio":
                self.decode_head = RogerioHead(
                    num_classes=self.head_cfg["num_classes"],
                    in_channels=self.head_cfg["num_channels"],
                    embed_dim=self.head_cfg["embed_dim"],
                    hidden_dim=self.head_cfg["hidden_dim"],
                    num_heads=self.head_cfg["num_heads"],
                    num_layers=self.head_cfg["num_layers"],
                    dropout=self.head_cfg["conv_dropout"],
                    num_frames=self.head_cfg["num_frames"],
                    init_super=True,
                )
            elif self.cfg["cls_head"] == "mlp":
                self.decode_head = MLPHead(
                    num_classes=self.head_cfg["num_classes"],
                    in_channels=self.head_cfg["num_channels"],
                    num_frames=self.head_cfg["num_frames"],
                    # embed_dim = self.head_cfg['embed_dim'],
                    dropout=self.head_cfg["linear_dropout"],
                    hidden_dim=self.head_cfg["hidden_dim"],
                )
            elif self.cfg["cls_head"] == "conv":
                self.decode_head = ConvHead(
                    num_classes=self.head_cfg["num_classes"],
                    in_channels=self.head_cfg["num_channels"],
                    dropout=self.head_cfg["conv_dropout"],
                )
            elif self.cfg["cls_head"] == "convmlp":
                self.decode_head = ConvMLP(
                    num_classes=self.head_cfg["num_classes"],
                    in_channels=self.head_cfg["num_channels"],
                    num_frames=self.head_cfg["num_frames"],
                    embed_dim=2048,
                    dropout=self.head_cfg["linear_dropout"],
                    hidden_dim=self.head_cfg["hidden_dim"],
                )
            else:
                raise Exception(f"Invalid decode head: {self.cfg['cls_head']}")

        if self.cfg["apply_batchnorm"]:
            # print(self.head_cfg['num_channels'])
            self.batchnorm = nn.BatchNorm2d(self.head_cfg["num_channels"])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        ## test for now fix the unet
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        textual_inversion_token_path = self.cfg["textual_inversion_token_path"]
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
        embedding = embedding.to(
            dtype=self.text_encoder.dtype, device=self.text_encoder.device
        )

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

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def setup(self, stage=None):
        self.decode_head.apply(self.init_weights)

    def initialize_model(self):
        pass

    def create_text_embeddings(
        self, latents, img_metas=None, captions=None, texts=None
    ):
        texts = []
        _cs = []
        conds = []
        if captions is None:
            for l in latents:
                text = "human"
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
        if self.cfg.get("use_scaled_encode", False):
            with torch.no_grad():
                first_stage = self.sd_model.encode_first_stage(img)
                latents = self.sd_model.get_first_stage_encoding(
                    self.sd_model.encode_first_stage(img)
                )
        else:
            with torch.no_grad():
                latents = self.encoder_vq.encode(img)
            latents = latents.mode().detach()

        c_crossattn = None
        if captions is not None:
            c_crossattn = self.create_text_embeddings(
                latents, img_metas=img_metas, captions=captions, texts=texts
            )
        elif "class_names" in self.cfg["text_conditioning"]:
            c_crossattn = self.class_names_embs.repeat(img.shape[0], 1, 1).to(
                img.device
            )

        t = torch.from_numpy(np.array([1])).to(img.device)

        outs = self.model(latents, t, c_crossattn=[c_crossattn])

        viz = False
        if viz:

            def plot_ca(ca):
                nrows = 3
                ncols = ca.shape[0] // nrows + 1
                fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5))
                for i, ax in enumerate(axes.flatten()):
                    if i < ca.shape[0]:
                        img = T.functional.resize(
                            ca[i].unsqueeze(0), (64, 64), antialias=True
                        )[0]
                        ax.imshow(img.detach().cpu().numpy(), cmap="gray")
                    ax.axis("off")
                plt.show()

            print(captions[0])
            plot_ca(outs[1][0][640:660])

        return outs

    def forward(self, x, img_metas=None, return_feat=False):
        if type(x) is list:
            x = torch.stack(x)
        # log first video of batch
        # tensor_to_video(x[0], fps=4)

        captions = None
        if hasattr(self, "blip_captions") and img_metas is not None:
            captions = [self.blip_captions[name] for name in img_metas]

        all_features = []

        # concatentate all video frames, then split into image batches
        frames_per_batch = self.cfg["diffusion_batch_size"]
        n_batch, n_frames, channels, height, width = x.shape
        x = x.reshape(n_batch * n_frames, channels, height, width)
        img_batches = torch.split(x, frames_per_batch)
        all_captions = None
        if captions is not None:
            all_captions = []
            for c in captions:
                all_captions.extend([c for _ in range(n_frames)])
        for i in range(len(img_batches)):
            img_batch = img_batches[i]
            caption_batch = None
            if all_captions is not None:
                caption_batch = all_captions[
                    i
                    * frames_per_batch : min(
                        i * frames_per_batch + frames_per_batch, len(all_captions)
                    )
                ]
            feat_layer_idx = 2
            # if self.head_cfg['num_channels'] < 640:
            #     feat_layer_idx = 0
            # elif self.head_cfg['num_channels'] < 1280:
            #     feat_layer_idx = 1
            features = self.extract_feat(img_batch, captions=caption_batch)[1]
            all_features.append(features)
        features = torch.cat(all_features)
        ret_features = features.reshape(n_batch, n_frames, *features.shape[1:])
        if self.cfg["use_only_attn"]:
            features = features[:, 640:, :, :]
        if self.cfg["apply_batchnorm"]:
            features = self.batchnorm(features)
        features = features.reshape(n_batch, n_frames, *features.shape[1:])

        if not self.use_decode_head:
            return features

        x = self.decode_head(features)

        if return_feat:
            return x, ret_features
        return x

    # def inference(self, img, captions=None):
    # x = [torchvision.transforms.ToTensor()(x) for x in img]

    # orig_images = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],tadpvid
    #                                                 std=[0.229, 0.224, 0.225])(x) for x in x]
    # orig_images = [datapoints.Image(x) for x in orig_images]

    # from torchvision.transforms import v2 as T
    # _size = 512
    # trans = T.Compose(
    #     [
    #         T.Resize((_size, _size)),
    #     ]
    # )
    # orig_images = trans(orig_images)

    # orig_images = torch.stack(orig_images)
    # orig_images = torchvision.models.detection.image_list.ImageList(orig_images,
    #                                                                 image_sizes=[(_size, _size)] * len(orig_images))

    # orig_images_tensors = orig_images.tensors.to(self.device)

    # features = self.extract_feat(orig_images_tensors, texts=captions)

    # x = self.decode_head(features[0])

    # return x

    def configure_optimizers(self):
        # TODO: double check here
        # have differernt learning rate for different layers
        # parameters to optimize
        lesslr_no_decay = list()
        lesslr_decay = list()
        no_lr = list()
        no_decay = list()
        decay = list()
        for name, m in self.named_parameters():
            if "unet" in name and "norm" in name:
                lesslr_no_decay.append(m)
            elif "unet" in name:
                lesslr_decay.append(m)
            elif "encoder_vq" in name:
                no_lr.append(m)
            elif "norm" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        if self.freeze_backbone:
            params_to_optimize = [
                {"params": lesslr_no_decay, "weight_decay": 0.0, "lr_scale": 0.0},
                {"params": lesslr_decay, "lr_scale": 0.0},
                {"params": no_lr, "lr_scale": 0.0},
                {"params": no_decay, "weight_decay": 0.0},
            ]
        else:
            params_to_optimize = [
                {"params": lesslr_no_decay, "weight_decay": 0.0, "lr_scale": 0.01},
                {"params": lesslr_decay, "lr_scale": 0.01},
                {"params": no_lr, "lr_scale": 0.0},
                {"params": no_decay, "weight_decay": 0.0},
                {"params": decay},
            ]
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=5e-4, weight_decay=1e-2, amsgrad=False
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda x: (
                1 - x / (self.cfg["dataset_len"] * self.cfg["max_epochs"])
            )
            ** 0.9,
        )

        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}
        ]

    def training_step(self, batch, batch_idx):
        try:
            x, y, metas = batch
        except ValueError:
            x, y = batch
            metas = None
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
        try:
            x, y, metas = batch
        except ValueError:
            x, y = batch
            metas = [""]
        preds, features = self(x, img_metas=metas, return_feat=True)
        loss = self.metric(preds, y)
        accuracy = self.accuracy(preds, y)
        self.log_dict({"test_loss": loss, "test_acc": accuracy}, sync_dist=True)

        # log ca for first video in each batch
        video, features, label, meta = (
            x[0],
            features[0][self.head_cfg["num_frames"] // 2],
            y[0],
            metas[0],
        )
        self._log_wandb_media(video, features, label, meta)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            x, y, metas = batch
        except ValueError:
            x, y = batch
            metas = [""]
        preds, features = self(x, img_metas=metas, return_feat=True)
        loss = self.metric(preds, y)
        accuracy = self.accuracy(preds, y)
        self.log_dict({"val_loss": loss, "val_acc": accuracy}, sync_dist=True)

        # log video and attention maps every 10 batches in each validation loop
        if batch_idx % 10 == 0:
            self._print_log(preds, y)
            video, features, label, meta = (
                x[0],
                features[0][self.head_cfg["num_frames"] // 2],
                y[0],
                metas[0],
            )
            self._log_wandb_media(video, features, label, meta)

    def _print_log(self, logits, labels):
        probs = F.softmax(logits, 1).detach().cpu().numpy()
        preds = torch.argmax(logits, 1).detach().cpu().numpy()
        pred_labels = (
            [pred for pred in preds]
            if len(self.class_names) < 51
            else [self.class_names[pred] for pred in preds]
        )
        print("\npredictions:", pred_labels)
        print("probabilities of preds:", [probs[i][c] for i, c in enumerate(preds)])
        print(
            "truth:",
            [self.class_names[truth] for truth in labels.detach().cpu().numpy()],
            "\n",
        )

    def _log_wandb_media(self, video, features, label, meta):
        """log diffusion cross attention maps from middle frame of each video"""

        num_channels = self.head_cfg["num_channels"]
        if num_channels < 640:
            num_feats = 320
        elif num_channels < 1280:
            num_feats = 640
        else:
            num_feats = 1280
        num_tokens = num_channels - num_feats

        features = T.functional.resize(features, (64, 64), antialias=True)
        feat_maps = features[:10]
        feat_maps = tensor_to_viridis_pil(feat_maps)
        ca_maps = features[num_feats:num_channels]
        ca_maps = tensor_to_viridis_pil(ca_maps)

        tokens = ["" for _ in range(num_tokens)]
        if hasattr(self, "blip_captions"):
            caption = self.blip_captions[meta]
            _, t = self.sd_model.get_learned_conditioning(caption, return_tokens=True)
            tokens = [
                self.sd_model.cond_stage_model.tokenizer.decode(ind) for ind in t[0]
            ]
        elif hasattr(self, "class_names"):
            tokens = self.class_names

        frames = video.cpu().numpy()

        if self.log_ca:
            self.logger.experiment.log(
                {
                    "video": wandb.Video(frames, caption=self.class_names[label]),
                    "cross_att_maps": [
                        wandb.Image(ca_map, caption=token)
                        for ca_map, token in zip(ca_maps, tokens)
                    ],
                    "features": [
                        wandb.Image(feature, caption=f"feature {i}")
                        for i, feature in enumerate(feat_maps)
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


def tensor_to_viridis_pil(x):
    normalize = lambda x: (x - x.min()) / (x.max() - x.min())
    arrs = x.detach().cpu().numpy()
    arrs = [normalize(arr) for arr in arrs]
    viridis = plt.cm.get_cmap("viridis")
    vir_arrs = [viridis(arr) for arr in arrs]
    vir_arrs = [(arr[:, :, :3] * 255).astype(np.uint8) for arr in vir_arrs]
    ca_maps = [Image.fromarray(arr) for arr in vir_arrs]
    return ca_maps
