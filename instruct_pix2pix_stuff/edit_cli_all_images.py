import sys
import math
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageOps
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as f

import einops
import safetensors
import k_diffusion as K
from einops import rearrange
from omegaconf import OmegaConf

from dataset import build_dataset
sys.path.append("./stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config
from textual_inversion import seed_everything
from textual_inversion_multiple_prompts import extract_placeholder_tokens


def get_args():
    parser = ArgumentParser()

    parser.add_argument("--gpu-id", default=0, type=int)

    parser.add_argument("--source-domain", default="cityscapes", type=str)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--placeholder-token", type=str)
    parser.add_argument("--initializer-token", type=str, default=None)
    parser.add_argument("--num_vectors", type=int, default=1)
    parser.add_argument("--placeholder-token-ckpt-path", required=True, type=str)

    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    return args


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
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


def load_placeholder_token_embedding(model, placeholder_tokens, placeholder_token_ckpt_path, num_vectors, initializer_token=None):
    device = next(model.parameters()).device

    tokenizer = model.cond_stage_model.tokenizer
    text_encoder = model.cond_stage_model.transformer

    if placeholder_token_ckpt_path.endswith('.safetensors'):
        ckpt = safetensors.torch.load_file(placeholder_token_ckpt_path)
    else:
        ckpt = torch.load(placeholder_token_ckpt_path)
    print(f'Loading the placeholder token weight from {placeholder_token_ckpt_path}')
    
    ckpt = {k: v.to(device) for k, v in ckpt.items()}
    assert set(ckpt.keys()) == set(placeholder_tokens), f'The placeholder tokens are not matched: {set(ckpt.keys())},  {set(placeholder_tokens)}'

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    placeholder_token_weight = text_encoder.get_input_embeddings().weight
    
    if list(ckpt.keys()) == ['placeholder_token']:
        # multiple prompts
        placeholder_token_weight.data[placeholder_token_ids] = ckpt['placeholder_token']
    else:
        # original single prompts
        placeholder_token_weight.data[placeholder_token_ids] = ckpt[placeholder_tokens[0]]


def main():
    args = get_args()
    print(args)
    seed_everything(args.seed)

    dataset = build_dataset(
        source_domain=args.source_domain,
        target_domain=None,
        resolution=args.resolution
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device(f'cuda:{args.gpu_id}')

    placeholder_tokens = extract_placeholder_tokens(args.edit)
    print('Placeholder tokens:', placeholder_tokens)

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().to(device)

    if len(placeholder_tokens) == 0:
        pass

    else:
        if len(placeholder_tokens) == 1:
            from textual_inversion import add_placeholder_token_to_model
            placeholder_token_ids = add_placeholder_token_to_model(model, args)

        else:
            from textual_inversion_multiple_prompts import add_placeholder_token_to_model
            placeholder_token_ids = add_placeholder_token_to_model(model, placeholder_tokens)

        load_placeholder_token_embedding(
            model,
            placeholder_tokens,
            args.placeholder_token_ckpt_path,
            args.num_vectors,
            args.initializer_token
        )

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)

    with torch.no_grad(), torch.autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]

        uncond = {}
        uncond["c_crossattn"] = [model.get_learned_conditioning([""])]

        for src_imgs, src_img_paths, ori_size in tqdm(data_loader, total=len(data_loader), desc='editting images...'):
            syn_img_path = Path(src_img_paths[0].replace(args.source_domain, args.output_dir))
            if syn_img_path.exists():
                continue
            
            src_imgs = src_imgs.to(device)
            encoded_src_imgs = model.encode_first_stage(src_imgs).mode()

            sigmas = model_wrap.get_sigmas(args.steps)

            cond["c_concat"] = [encoded_src_imgs]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": args.cfg_text,
                "image_cfg_scale": args.cfg_image,
            }
            torch.manual_seed(args.seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args, disable=True)
            x = model.decode_first_stage(z)
            
            syn_img = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0).squeeze()  # 0~1
            syn_pil_img = f.to_pil_image(syn_img)  # *255
            ori_size = [x.item() for x in ori_size]
            syn_pil_img = syn_pil_img.resize(ori_size)

            syn_img_dir_path = syn_img_path.parents[0]
            syn_img_dir_path.mkdir(parents=True, exist_ok=True)
            syn_pil_img.save(str(syn_img_path))
            tqdm.write(f'Saved to {str(syn_img_path)}')

    # TODO: soft link

if __name__ == "__main__":
    main()
