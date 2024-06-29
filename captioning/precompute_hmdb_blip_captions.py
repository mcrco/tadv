import argparse
import numpy as np
import torch
import os

import yaml
from PIL import Image
import json
from tqdm import tqdm

# from gpt_interface import GPTInterface
from captioning.blip import BLIPInterface

from gpt_interface import GPTInterface
from datasets.datamodules import PascalVOCDataModule


class BlipDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def get_image(self, idx, return_img_file=False):
        img_file = self.img_paths[idx]

        image = Image.open(img_file).convert("RGB")

        ret_list = [image]

        if return_img_file:
            ret_list.append(img_file)

        return ret_list

def get_files_recursively(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def parse_args():
    parser = argparse.ArgumentParser(description='Caption a dataset using BLIP2')
    parser.add_argument('--dataset', help='dataset to caption')
    parser.add_argument('--max_new_tokens', type=int, default=77, help='max number of tokens generated')
    parser.add_argument('--min_new_tokens', type=int, default=0, help='min number of tokens generated')
    parser.add_argument('--frames_per_vid', type=int, default=10, help='number of frames to caption per video')
    parser.add_argument('--params_name', type=str, default=None, help='params string (added to end of file name)')
    parser.add_argument('--do_blip', action='store_true', default=False)
    parser.add_argument('--do_gpt_summary', action='store_true', default=False)

    # Blip Generation parameters

    return parser.parse_args()


def main():
    args = parse_args()

    frame_caption_file = f"{args.dataset}_{args.params_name}_captions.json" 
    frame_root = f'data/{args.dataset}/frames/'

    def get_frame_paths(video):
        vid_path = os.path.join(frame_root, video)
        num_frames = len(os.listdir(vid_path))
        frame_indices = np.linspace(0, num_frames - 3, args.frames_per_vid).astype(int)
        frame_files = [str(i) + '.jpg' for i in frame_indices]
        return [os.path.join(vid_path, file) for file in frame_files]

    if args.do_blip:
        # parse blip_generation_parameters
        blip_generation_dict = {
            'max_new_tokens': args.max_new_tokens,
            'min_new_tokens': args.min_new_tokens,
        }

        all_frame_paths = []
        for vid in os.listdir(frame_root):
            all_frame_paths.extend(get_frame_paths(vid))

        img_name_dict = {}
        for frame_path in all_frame_paths:
            img_name_dict[frame_path] = frame_path

        dataset = BlipDataset(all_frame_paths)
        bi = BLIPInterface(dataset, args.dataset, args.params_name, blip_generation_dict)
        all_frame_captions = bi(img_name_dict=img_name_dict, profiling=True)

        video_frame_captions = {}
        for vid in os.listdir(frame_root):
            frame_paths = get_frame_paths(vid)
            video_captions = [all_frame_captions[path]['captions'][0] for path in frame_paths]
            video_frame_captions[vid] = video_captions

        with open(os.path.join('./captions/', frame_caption_file), 'w') as f:
            json.dump(video_frame_captions, f)

    if args.do_gpt_summary:
        cfg = yaml.load(open("./captioning/gpt_cfg.yaml", "r"), Loader=yaml.FullLoader)
        gpt = GPTInterface(cfg)
        with open(os.path.join('./captions/', frame_caption_file)) as f:
            captions = json.load(f)

        prompt = "summarize what might be happening in a video given the following sequential frame captions, \
                  but do not mention anything about the video itself, such as starting the summary with 'this video shows' or the like: "

        video_captions = {}
        for vid in tqdm(os.listdir(frame_root)):
            gpt_batch = captions[vid]
            gpt_batch = ', '.join(gpt_batch)
            _prompt = prompt + gpt_batch
            video_captions[vid] = gpt.general_gpt_task(_prompt)

        out_path = './captions'
        os.makedirs(out_path, exist_ok=True)
        with open(os.path.join(out_path, args.dataset + '_captions.json'), 'w') as f:
            json.dump(video_captions, f)

if __name__ == '__main__':
    main()
