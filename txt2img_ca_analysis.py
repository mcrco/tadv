import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import matplotlib.pyplot as plt

from ldm_cross_attention.util import instantiate_from_config
from ldm_cross_attention.models.diffusion.ddim import DDIMSampler
from ldm_cross_attention.models.diffusion.plms import PLMSSampler
import torchvision
from ldm_cross_attention.ca_analysis_utils import TextConditioningWrapper
from datasets.VOCDataset import classes
import json


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="cross_attention_analysis/"
    )
    parser.add_argument(
        "--make_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ldm_cross_attention/v1-cross_attention_inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    ## ADDED PARAMS FOR CROSS ATTENTION VISUALIZATION

    # cross attention plotting parameters
    parser.add_argument("--visualize_cross_attention", action='store_true', default=False)

    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--timesteps_to_visualize', type=str, default=None,
                        help='by default all timesteps in the step_range will be visualized, but a comma separated list of timesteps can be provided to reduce the number of visualizations')
    parser.add_argument('--include_head_average', action='store_true', default=False, help='include the average of the heads in the visualization')
    parser.add_argument('--step_range', type=str, default='0,1', help='the range of timesteps for diffusion (0-50), one step will not change the actual image much but produces meaningful attention maps')

    parser.add_argument('--only_save_summary', action='store_true', default=False, help='only save the summary figure')
    parser.add_argument('--include_eos', action='store_true', default=False, help='visualize eos token')
    parser.add_argument("--save_to_numpy", action='store_true', default=False)

    # text conditioning wrapper
    parser.add_argument('--text_conditioning', type=str, default=None, help='type of text conditioning')
    parser.add_argument("--caption_path", type=str, default=None)
    parser.add_argument('--class_embedding_path', type=str, default='./TADP/vpd/pascal_class_embeddings.pth')

    opt = parser.parse_args()

    seed_everything(opt.seed)
    if opt.step_range is not None:
        opt.step_range = [int(t) for t in opt.step_range.split(',')]
    # process timesteps to visualize

    if opt.timesteps_to_visualize is not None:
        opt.timesteps_to_visualize = [int(t) for t in opt.timesteps_to_visualize.split(',')]
    else:
        time_step_opts = [1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241,
                          261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501,
                          521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761,
                          781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981]
        opt.timesteps_to_visualize = time_step_opts[opt.step_range[0]:opt.step_range[1]]

    config = OmegaConf.load(f"{opt.config}")
    config.model.params.unet_config.params.visualize_ca = opt.visualize_cross_attention
    if opt.text_conditioning == 'class_emb':
        name = '/'.join(opt.class_embedding_path.split('/')[1:]).replace('.pth', '').replace('/', '_')
        outpath = os.path.join(opt.outdir, name)
    elif opt.text_conditioning == 'class_names':
        outpath = os.path.join(opt.outdir, 'class_names')
    else:
        outpath = os.path.join(opt.outdir, opt.caption_path.split('/')[-1].replace('.json', ''))
    config.model.params.unet_config.params.visualize_ca_params = {
        'visualize_specific_timesteps': opt.timesteps_to_visualize,
        'include_head_average': opt.include_head_average,
        'output_folder': outpath,
        'only_save_summary': opt.only_save_summary,
        'save_to_numpy': opt.save_to_numpy}
    model = load_model_from_config(config, f"{opt.ckpt}")
    # print(type(model))
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(outpath, exist_ok=True)

    batch_size = opt.batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = 0

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

    def load_image(path):
        x_samples_ddim = Image.open(path)
        x_samples_ddim = torchvision.transforms.Resize((512, 512))(x_samples_ddim)
        x_samples_ddim = torch.tensor(np.array(x_samples_ddim)).permute(2, 0, 1).unsqueeze(0) / 255
        #plot_image(x_samples_ddim)
        x_samples_ddim = x_samples_ddim.to('cuda')
        first_stage = model.encode_first_stage(x_samples_ddim)
        code = model.get_first_stage_encoding(model.encode_first_stage(x_samples_ddim))
        return code

    cfg = {'text_conditioning': opt.text_conditioning,
           'class_names': classes,
           'caption_path': opt.caption_path
           }
    tcw = TextConditioningWrapper(cfg, model, class_embedding_path=opt.class_embedding_path)

    # pascal_img_path = 'data/VOCdevkit/VOC2012/JPEGImages/'
    # img_id_file = 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt'
    # with open(img_id_file, 'r') as f:
    #     img_ids = f.readlines()
    # img_ids = [img_id.strip() for img_id in img_ids]

    # img_ids = list(img_ids)
    # img_ids = img_ids[:opt.n_samples]

    # this is the id for the main image in the paper, uncomment to append to list of images to visualize
    # img_ids.append('2010_001715')

    # batchify
    root_img_path = 'ca_test'
    img_ids = ['tadp']
    img_id_batches = list(chunk(img_ids, batch_size))

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for img_id_batch in tqdm(img_id_batches, desc="data"):
                    uc = None

                    # encode the text and store the byte pair decodings
                    c, tokens = tcw.create_text_embeddings(img_metas={'img_id': list(img_id_batch)})
                    if opt.text_conditioning != 'class_emb' and opt.text_conditioning != 'class_names':
                        prompts = []
                        for img_id in img_id_batch:
                            prompts.append(tcw.blip_captions[img_id]['captions'])
                        bpe_decoded_prompts = [
                            [model.cond_stage_model.tokenizer.decode(tokens[j][i]) for i in range(77)] for j in
                            range(len(prompts))]
                    elif opt.text_conditioning == 'class_names':
                        bpe_decoded_prompts = [
                            [model.cond_stage_model.tokenizer.decode(tokens[j][i]) for i in range(24)] for j in
                            range(len(img_id_batch))]
                    else:
                        bpe_decoded_prompts = [classes] * batch_size

                    # add batch information to the cross attention target index dict
                    for key in model.model.diffusion_model.cross_attention_target_index_dict:
                        key.plot_dict['sample_names'] = img_id_batch
                        key.plot_dict['include_eos'] = opt.include_eos  # TODO check if can be moved up

                    start_codes = []
                    for ii, img_id in enumerate(img_id_batch):
                        start_codes.append(load_image(f'{root_img_path}/{img_id}.jpg'))
                    start_codes = torch.cat(start_codes, dim=0)

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=min(batch_size, len(start_codes)),
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_codes,
                                                     # pass bpe decoding for naming each token
                                                     prompts=bpe_decoded_prompts,
                                                     tokens=tokens,
                                                     step_range=opt.step_range
                                                     )

                    if not opt.skip_save:
                        for img_id in img_id_batch:
                            img_path = os.path.join(root_img_path, f'{img_id}.jpg')
                            if not os.path.exists(os.path.join(sample_path, f'{img_id}.jpg')):
                                os.symlink(img_path, os.path.join(sample_path, f'{img_id}.jpg'))
                            base_count += 1

                        # for x_sample in x_samples_ddim:
                        #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        #     Image.fromarray(x_sample.astype(np.uint8)).save(
                        #         os.path.join(sample_path, f"{base_count:05}.jpg"))
                        #     base_count += 1


    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    # this script generates cross attention visualizations for images in the pascal VOC dataset
    main()
