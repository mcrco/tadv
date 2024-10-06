import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torchvision.io import read_video

from TADV.t2v_utils.attn_control import AttentionController, register_attn_hook
from TADV.t2v_utils.svd_extract import pipe_features
from TADV.t2v_utils.unet_extract import register_unet_features


def preprocess_video(path):
    video = read_video(path, output_format="TCHW")[0]
    video = video.float().half()
    video = video.to("cuda")
    video = T2.UniformTemporalSubsample(8)(video)
    video = torch.stack([T.Resize((384, 512))(frame) for frame in video])
    video = T.Normalize(mean=(128,), std=(128,))(video)
    return video


def get_latents(pipe, frames):
    with torch.no_grad():
        latents = pipe.vae.encode(frames).latent_dist.mode()
    return latents


def decode_latents(pipe, latents):
    decode_output = pipe.vae.decode(latents).sample
    video = T.Normalize(mean=(-1,), std=(1 / 128))(decode_output)
    video = video.float().permute(0, 2, 3, 1).detach().cpu().numpy().astype(int)
    return video


if __name__ == "__main__":
    pipe = DiffusionPipeline.from_pretrained(
        "ali-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
    )
    pipe = pipe.to("cuda")
    attn_controller = AttentionController()
    attn_controller.reset()
    register_attn_hook(pipe.unet, attn_controller)
    register_unet_features(pipe.unet)

    video = preprocess_video("./videos/golf.avi")
    latents = get_latents(pipe, video)
    latents = latents.permute(1, 0, 2, 3).unsqueeze(0)
    prompt = "person swinging golf club"
    features = pipe_features(
        pipe=pipe, prompt=prompt, latents=latents, guidance_scale=0
    )
    # cross_attns, temp_attns = attn_controller.process_attns()
    cross_attns, temp_attns = attn_controller.get_avg_attns()
    prompt_tokens = pipe.tokenizer.batch_decode(
        pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    )

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        frame = (
            T.Normalize(mean=(-1), std=(1 / 128))(video)[i].permute(1, 2, 0).float()
            / 255
        )
        ax.imshow(frame.detach().cpu())
        ax.axis("off")

    fig, axes = plt.subplots(4, 4, figsize=(10, 5))
    for r in range(4):
        for c in range(4):
            attn = T.functional.resize(
                cross_attns[768][r][c + 1].reshape(24, 32).unsqueeze(0),
                (60, 80),
                antialias=True,
            )[0]
            im = axes[r][c].imshow(attn.detach().cpu(), cmap="viridis")
            axes[r][c].set_title(prompt_tokens[c + 1])
            axes[r][c].axis("off")
            # fig.colorbar(im, ax=axes[r][c])

    fig, axes = plt.subplots(8, 8, figsize=(10, 5))
    for r in range(8):
        for c in range(8):
            attn = T.functional.resize(
                temp_attns[768][r][c].reshape(24, 32).unsqueeze(0),
                (60, 80),
                antialias=True,
            )[0]
            axes[r][c].imshow(attn.detach().cpu(), cmap="viridis")
            axes[r][c].axis("off")

    plt.tight_layout()
    plt.show()
