import argparse
import os

import torch
import torch.nn.functional as F
import yaml
import torchvision
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
import matplotlib.pyplot as plt
import ldm

from TADP.tadp_video import TADPVid


parser = argparse.ArgumentParser()
parser.add_argument("--val_dataset_name", default="pascal", type=str)
parser.add_argument("--cross_domain_target", default="watercolor", type=str)
parser.add_argument('--cross_blip_caption_path', type=str, default=None,
                    help='path to cross blip captions')
parser.add_argument('--dreambooth_checkpoint', type=str, default=None, help='path to dreambooth checkpoint')
parser.add_argument('--textual_inversion_token_path', type=str, default=None,
                    help='path to textual inversion token path')

# experiment parameters
parser.add_argument("--model_name", type=str, default="DeeplabV3Plus")
parser.add_argument("--from_scratch", action='store_true', default=False)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument('--eval_dataset', type=str, nargs='+', default=['pascal'])
parser.add_argument('--optimizer_config_preset', type=int, default=0)
parser.add_argument('--strategy', type=str, default='')
parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument('--freeze_backbone', type=int, default=0)
parser.add_argument('--log_ca', action='store_true', default=False)
parser.add_argument('--freeze_batchnorm', type=int, default=0)
parser.add_argument('--gradient_clip_val', type=float, default=10.0)
parser.add_argument('--apply_batchnorm', type=int, default=1)

parser.add_argument("--accum_grad_batches", type=int, default=1)
parser.add_argument("--freeze_text_adapter", type=int, default=1)

parser.add_argument('--train_dataset', type=str, nargs='+', default=['VOC2012_ext'])
parser.add_argument('--train_max_samples', type=int, default=None)

# TADP specific parameters
parser.add_argument('--text_conditioning', type=str, default='class_names')
parser.add_argument('--min_blip', type=int, default=0)
parser.add_argument('--task_inversion_lr', type=float, default=0.002)
parser.add_argument('--use_scaled_encode', action='store_true', default=False)
parser.add_argument('--append_self_attention', action='store_true', default=False)
parser.add_argument('--use_decoder_features', action='store_true', default=False)
parser.add_argument('--use_text_adapter', action='store_true', default=False)
parser.add_argument('--cond_stage_trainable', action='store_true', default=False)
parser.add_argument('--blip_caption_path', type=str, default=None)
parser.add_argument('--no_attn', action='store_true', default=False)
parser.add_argument('--use_only_attn', action='store_true', default=False)
parser.add_argument('--present_class_embeds_only', action='store_true', default=False)
parser.add_argument('--cls_head', type=str, default='mlp')
parser.add_argument('--diffusion_batch_size', type=int, default=1)
parser.add_argument('--normalize_videos', action='store_true', default=True)

parser.add_argument('--trainer_ckpt_path', type=str, default=None)
parser.add_argument('--save_checkpoint_path', type=str, default='out/')
parser.add_argument('--train_debug', action='store_true', default=False)
args = parser.parse_args()

cfg = yaml.load(open("./sd_tune.yaml", "r"), Loader=yaml.FullLoader)
cfg["stable_diffusion"]["use_diffusion"] = True
cfg["max_epochs"] = 50
cfg["dataset_len"] = 1
cfg["freeze_text_adapter"] = True

if args.no_attn and args.use_only_attn:
    raise ValueError('Cannot use both no_attn and use_only_attn')

model_kwargs = {}
cfg['text_conditioning'] = args.text_conditioning
cfg['use_scaled_encode'] = args.use_scaled_encode
cfg['class_names'] = ['bird', 'dog', 'human', 'background']
cfg['append_self_attention'] = args.append_self_attention
cfg['use_text_adapter'] = args.use_text_adapter
cfg['cond_stage_trainable'] = False
if args.append_self_attention:
    model_kwargs['unet_config'] = {'attn_selector': 'up_cross+down_cross-up_self+down_self'}
model_kwargs['unet_config'] = {'use_attn': not args.no_attn}  # default for use_attn ends up true
cfg['use_attn'] = not args.no_attn
cfg['use_only_attn'] = args.use_only_attn
cfg['use_decoder_features'] = False
cfg['use_token_embeds'] = False
cfg['present_class_embeds_only'] = args.present_class_embeds_only

cfg['dreambooth_checkpoint'] = args.dreambooth_checkpoint
cfg['textual_inversion_token_path'] = args.textual_inversion_token_path
cfg['val_dataset_name'] = args.val_dataset_name
cfg['cross_blip_caption_path'] = args.cross_blip_caption_path
cfg['cls_head'] = args.cls_head
cfg['diffusion_batch_size'] = args.diffusion_batch_size
cfg['normalize_videos'] = args.normalize_videos
cfg['apply_batchnorm'] = args.apply_batchnorm

head_cfg = {
    'num_classes': 51,
    'num_channels': 717,
    'embed_dim': 512,
    'hidden_dim': 2048,
    'num_heads': 8,
    'num_layers': 6,
    'dropout': 0.1,
    'num_frames': 8,
    'init_super': True
}

model = TADPVid(cfg=cfg, class_names=cfg['class_names'], freeze_backbone=args.freeze_backbone, log_ca=args.log_ca, head_cfg=head_cfg)
model.to('cuda')

frame_resize = T.Resize((512, 512))
norm_fn = lambda x: x / 255.0

def plot_tensor_value_distributions(tensor):
    """
    Plots the value distributions of a tensor with shape (1, 3, H, W).

    Args:
    tensor (torch.Tensor): A tensor with shape (1, 3, H, W) where 1 is the batch dimension,
                           3 is the number of channels (e.g., RGB), and H, W are height and width.
    """
    if tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError("Expected tensor shape (1, 3, H, W).")

    # Remove the batch dimension (1) by squeezing the tensor
    tensor = tensor.squeeze(0)  # Now the shape is (3, H, W)

    # Plot value distributions for each channel
    channels = ['Red', 'Green', 'Blue']

    plt.figure(figsize=(15, 5))

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(tensor[i].flatten().numpy(), bins=50, color=channels[i].lower())
        plt.title(f'{channels[i]} Channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def plot_image(tensor, title=None):
    """
    Plots an image from a tensor.

    Args:
    tensor (torch.Tensor): A tensor representing an image. The tensor can be in the shape
                           (C, H, W) or (1, C, H, W) where C is the number of channels,
                           H is the height, and W is the width.
    title (str, optional): The title for the plot.
    """
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)  # Remove batch dimension if present

    if tensor.size(0) == 1:  # Grayscale image (1, H, W)
        plt.imshow(tensor.squeeze(0), cmap='gray')
    elif tensor.size(0) == 3:  # RGB image (3, H, W)
        plt.imshow(tensor.permute(1, 2, 0))  # Change to (H, W, C) for plotting
    else:
        raise ValueError("Expected tensor with shape (C, H, W) where C is 1 or 3.")

    if title:
        plt.title(title)

    plt.axis('off')  # Hide the axes
    plt.show()

def preprocess_img(path):
    # img = torchvision.io.read_image(os.path.join('ca_test/', path))
    # img = img.float().cuda()
    # img = norm_fn(frame_resize(img))
    # img = img.unsqueeze(0)
    # plot_image(img.cpu())

    x_samples_ddim = Image.open(path)
    x_samples_ddim = torchvision.transforms.Resize((512, 512))(x_samples_ddim)
    x_samples_ddim = torch.tensor(np.array(x_samples_ddim)).permute(2, 0, 1).unsqueeze(0) / 255
    #plot_image(x_samples_ddim)
    img = x_samples_ddim.to('cuda')
    # code = model.get_first_stage_encoding(model.encode_first_stage(x_samples_ddim))

    return img

def plot_ca(ca, tokens):
    nrows = 3
    ncols = ca.shape[0] // nrows + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5))
    for i, ca in enumerate(ca):
        axes[i // ncols][i % ncols].imshow(ca.detach().cpu().numpy(), cmap='viridis')
        axes[i // ncols][i % ncols].set_title(tokens[i])
        axes[i // ncols][i % ncols].axis('off')
    for r in range(nrows):
        for c in range(ncols):
            axes[r][c].axis('off')
    plt.show()

tadp_img = preprocess_img('ca_test/abs.jpg')
# captions = [' '.join(cfg['class_names'])]
captions = ['A man and woman are consistently doing an exercise routine on a television, with a focus on side crunches.']
tadp_features = model.extract_feat(tadp_img, captions=captions)
# for feat in tadp_features:
#     print(feat.shape)
resize = lambda x : T.functional.resize(x, (64, 64), antialias=True)
tadp_features[1] = resize(tadp_features[1])
tadp_features[2] = resize(tadp_features[2])
tadp_ca = tadp_features[1][0][640:660]
# print(tadp_ca.min())
# print(tadp_ca.max())
# print(tadp_ca)
_, t = model.sd_model.get_learned_conditioning(captions[0], return_tokens=True)
tokens = [model.sd_model.cond_stage_model.tokenizer.decode(ind) for ind in t[0]]
plot_ca(tadp_ca, tokens)

# dog_img = preprocess_img('./dog.jpg')
# captions = [' '.join(cfg['class_names'])]
# dog_features = model.extract_feat(dog_img, captions=captions)
# dog_ca = dog_features[1][0][640:]
# print(dog_ca.min())
# print(dog_ca.max())
# print(dog_ca)
# plot_ca(dog_ca)

# bird_img = preprocess_img('./bird.jpg')
# birdfeatures = model.extract_feat(bird_img, captions=captions)
# birdca = birdfeatures[1][0][640:]
# plot_ca(birdca)

# both_img = preprocess_img('./both.jpg')
# both_features = model.extract_feat(both_img, captions=captions)
# both_ca = both_features[1][0][640:]
# plot_ca(both_ca)

# black_img = preprocess_img('./black.jpg')
# black_features = model.extract_feat(black_img, captions=captions)
# black_ca = black_features[1][0][640:]
# plot_ca(black_ca)
