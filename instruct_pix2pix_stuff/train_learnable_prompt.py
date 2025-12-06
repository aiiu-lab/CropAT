import os
import sys
import math
import random
import numpy as np
from pathlib import Path
from tqdm.auto import trange
from PIL import Image, ImageOps
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.utils import make_grid

import lpips
import einops
import safetensors
import k_diffusion as K
from einops import rearrange
from omegaconf import OmegaConf

from dataset import build_dataset
sys.path.append("./stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config
from clip_loss import ClipCosSimilarity


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu-id", default=0, type=int)

    parser.add_argument("--source-domain", default="cityscapes", type=str)
    parser.add_argument("--target-domain", default="cityscapes_foggy", type=str)

    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--placeholder-token", required=True, type=str)
    parser.add_argument("--initializer-token", type=str, default=None)
    parser.add_argument("--num_vectors", type=int, default=1)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--train-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--visualization-period", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500, help="Save learned_embeds.bin every X updates steps.",)
    parser.add_argument("--no-safe-serialization", action="store_true",  help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",)

    args = parser.parse_args()

    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model  # model_wrap

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        # 'cond': {
        #     'c_crossattn': instruction embedding
        # }
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)  # call `LatentDiffusion.apply_model()`
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def add_placeholder_token_to_model(model, args):
    # model:ldm.models.diffusion.ddpm_edit.LatentDiffusion
    # model.cond_stage_model.tokenizer: CLIPTokenizer
    # model.cond_stage_model.transformer: CLIPTextModel
    # model.first_stage_model: ldm.models.autoencoder.AutoencoderKL

    tokenizer = model.cond_stage_model.tokenizer
    text_encoder = model.cond_stage_model.transformer

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens = [args.placeholder_token] + additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the placeholder_token to ids
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))  # random initialization

    if 'initializer_token' in args and args.initializer_token is not None:
        init_token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        if len(init_token_ids) != len(placeholder_token_ids):
            raise ValueError('The number of the initializer token must be equal to that of the placeholder token.')
        
        token_embeds = text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for placeholder_token_id, init_token_id in zip(placeholder_token_ids, init_token_ids):
                token_embeds[placeholder_token_id] = token_embeds[init_token_id].clone()

    return placeholder_token_ids


def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    # model: model_wrap_cfg
    # extra_args: {
    #     'cond': {
    #         'c_crossattn': instruction embedding
    #     }
    # }
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = K.sampling.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = K.sampling.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = K.sampling.to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def main():
    args = get_args()
    seed_everything(args.seed)
    
    writer = SummaryWriter(args.output_dir)

    dataset = build_dataset(
        args.source_domain,
        args.target_domain,
        image_set='train',
        resolution=args.resolution
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    device = torch.device(f'cuda:{args.gpu_id}')

    assert args.placeholder_token is not None and args.placeholder_token != ""
    assert args.placeholder_token in args.edit
    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    config = OmegaConf.load(args.config)
    config.model.params.unet_config.params.use_checkpoint = args.use_checkpoint  # NOTE: set it to False forcely
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)  # ldm.models.diffusion.ddpm_edit.LatentDiffusion
    placeholder_token_ids = add_placeholder_token_to_model(model, args)
    model.eval().to(device)

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)

    # Freeze model while set the embeddings trainable
    model.requires_grad_(False)
    tokenizer = model.cond_stage_model.tokenizer
    text_encoder = model.cond_stage_model.transformer
    token_weight = text_encoder.get_input_embeddings().weight
    token_weight.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [token_weight],  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    loss_fn = lpips.LPIPS(net='vgg').to(device)

    # clone the original token embeddings
    orig_token_weight = token_weight.data.clone()
    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
    
    train_flag = True
    current_iters = 0
    optimizer.zero_grad()
    
    while train_flag:
        for src_img, tgt_img, src_img_path, tgt_img_path in data_loader:
            with torch.autocast('cuda'):
                src_img = src_img.to(device)  # (bs, 3, 256, 512)  -1~1
                tgt_img = tgt_img.to(device)  # (bs, 3, 256, 512)  -1~1

                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]  # [(1, 77, 768)] # differentiable

                with torch.no_grad():
                    cond["c_concat"] = [model.first_stage_model.encode(src_img).mode()]  # [(1, 4, 64, 64)] # `.mode()` return mean

                uncond = {}
                uncond["c_crossattn"] = [model.get_learned_conditioning([""])]  # (1, 77, 768)]
                uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                sigmas = model_wrap.get_sigmas(args.steps)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": args.cfg_text,
                    "image_cfg_scale": args.cfg_image,
                }
                torch.manual_seed(args.seed)
                z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                z = sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args, disable=True)
                x = model.differentiable_decode_first_stage(z)  # -1~1
                x = resize(x, tgt_img.shape[-2:])

                loss = loss_fn(x, tgt_img) / args.gradient_accumulation_steps
                
            loss.backward()
            if (current_iters + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            writer.add_scalar('loss', loss, global_step=current_iters)

            if current_iters == 0 or (current_iters + 1) % args.visualization_period == 0:
                src_img = resize(src_img, tgt_img.shape[-2:])
                src_img = torch.clamp((src_img + 1.0) / 2.0, min=0.0, max=1.0)
                tgt_img = torch.clamp((tgt_img + 1.0) / 2.0, min=0.0, max=1.0)
                syn_img = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)

                imgs = torch.cat([src_img, syn_img, tgt_img], dim=0)
                writer.add_images('src_syn_tgt_img', imgs, global_step=current_iters)
                writer.add_text('img_path/src', src_img_path[0], global_step=current_iters)
                writer.add_text('img_path/tgt', tgt_img_path[0], global_step=current_iters)

                save_dir = Path(args.output_dir) / 'images'
                if not save_dir.exists():
                    save_dir.mkdir(exist_ok=True, parents=True)
                save_path = save_dir / f'src_syn_tgt_img_{current_iters}.png'
                pil_img = to_pil_image(make_grid(imgs).cpu())
                pil_img.save(str(save_path))

                placeholder_token_weight = token_weight[~index_no_updates]
                writer.add_histogram(f'{args.placeholder_token}_weights', placeholder_token_weight, global_step=current_iters)
                writer.add_scalar(f'{args.placeholder_token}_weights/00', placeholder_token_weight[0, 0], global_step=current_iters)

            with torch.no_grad():
                token_weight[index_no_updates] = orig_token_weight[index_no_updates]

            current_iters += 1

            if (current_iters + 1) % args.save_steps == 0:
                save_dir = Path(args.output_dir) / 'checkpoint'
                if not save_dir.exists():
                    save_dir.mkdir(exist_ok=True, parents=True)

                save_dict = {args.placeholder_token: token_weight[~index_no_updates].detach().cpu()}
                
                if args.no_safe_serialization:
                    save_path = save_dir / f'placeholder_token_steps_{current_iters}.pt'
                    torch.save(str(save_dict), save_path)
                else:
                    save_path = save_dir / f'placeholder_token_steps_{current_iters}.safetensors'
                    safetensors.torch.save_file(save_dict, str(save_path), metadata={"format": "pt"})

            if current_iters == args.train_iters:
                train_flag = False
                break

    writer.close()


if __name__ == "__main__":
    main()
