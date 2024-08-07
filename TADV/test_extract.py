import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torchvision.io import read_video
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
from svd_extract import pipe_features
import torch
import matplotlib.pyplot as plt

def preprocess_video(path):
    video = read_video(path, output_format='TCHW')[0]
    video = video.float().half()
    video = video.to('cuda')
    video = T2.UniformTemporalSubsample(8)(video)
    # video = T.Normalize(mean = [0.485 * 256, 0.456 * 256, 0.406 * 256], std = [0.229 * 256, 0.224 * 256, 0.225 * 256])(video)
    video = T.Normalize(mean = (128,), std = (128,))(video)
    return video

def get_latents(pipe, frames):
    with torch.no_grad():
        latents = pipe.vae.encode(frames).latent_dist.sample()
    return latents

def decode_latents(pipe, latents):
    decode_output = pipe.vae.decode(latents).sample
    video = T.Normalize(mean = (-1,), std = (1/128))(decode_output)
    video = video.float().permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()
    return video

def extract(pipe, latents):
    features, cross_attns, temp_attns = pipe_features(
        pipe=pipe,
        prompt='baseball player',
        latents=latents
    )
    return features, cross_attns, temp_attns

if __name__ == '__main__':
    pipe = DiffusionPipeline.from_pretrained("ali-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to('cuda')
    video = preprocess_video('./videos/baseball.avi')

    latents = get_latents(pipe, video)
    latents = latents.permute(1, 0, 2, 3).unsqueeze(0)
    features, cross_attns, temp_attns = extract(pipe, latents)

    print([feat.shape for feat in features])
    print([attn.shape for attn in cross_attns])
    print([attn.shape for attn in temp_attns])

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        # ax.imshow(cross_attns[0][i][2].detach().cpu())
        ax.imshow(features[0][i][0].detach().cpu())
        ax.axis('off')  # Hide the axes

    plt.tight_layout()
    plt.show()

