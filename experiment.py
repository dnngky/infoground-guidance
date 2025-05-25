# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
# pylint: disable=unspecified-encoding
"""
Experiment script.
"""
import json
import os
import os.path as osp
import shutil
from argparse import ArgumentParser

import numpy as np
import PIL.Image as PImage
import torch
from tqdm import tqdm

from utils.misc import create_npz_from_sample_folder
from models import build_vae_var, VAR

# disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda _: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda _: None)

DEVICE = "cuda:0"

# allow 32-bit matmul to reduce GPU memory overhead
USE_TF32 = True
torch.backends.cudnn.allow_tf32 = USE_TF32
torch.backends.cuda.matmul.allow_tf32 = USE_TF32
torch.set_float32_matmul_precision("high" if USE_TF32 else "highest")

# set seed for reproducibility
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEPTHS = (16, 20, 24, 30, 36)

PATCH_NUMS_256 = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
PATCH_NUMS_512 = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)

NUM_CLASSES = 1000
NUM_SAMPLES_PER_CLASS = 50


def load_model(depth: int) -> VAR:
    """
    Load a VAR model with given depth.
    
    :param depth: Depth of VAR model to be loaded
    :type depth: int
    
    :return: Loaded VAR model
    :rtype: VAR
    """
    if depth not in DEPTHS:
        raise ValueError("Invalid depth")

    if not os.path.exists("./state_dicts"):
        os.makedirs("./state_dicts", exist_ok=False)

    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt = "state_dicts/vae_ch160v4096z32.pth"
    var_ckpt = f"state_dicts/var_d{depth}.pth"

    if not osp.exists(vae_ckpt):
        vae_ckpt_old = vae_ckpt.removeprefix("state_dicts/")
        os.system(f'wget {hf_home}/{vae_ckpt_old}')
        os.rename(vae_ckpt_old, vae_ckpt)
    if not osp.exists(var_ckpt):
        var_ckpt_old = var_ckpt.removeprefix("state_dicts/")
        os.system(f'wget {hf_home}/{var_ckpt_old}')
        os.rename(var_ckpt_old, var_ckpt)

    # build vae and var
    patch_nums = PATCH_NUMS_512 if depth == 36 else PATCH_NUMS_256
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=DEVICE, patch_nums=patch_nums,
        num_classes=NUM_CLASSES, depth=depth, shared_aln=(depth == 36),
    )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval()
    var.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    for param in var.parameters():
        param.requires_grad_(False)

    return var


def sample(
    var: VAR,
    label: int,
    batch_size: int,
    w_cfg: float | None,
    w_igg: float | None
) -> np.ndarray:
    """
    Sample a batch of images from given VAR model.
    
    :param var: VAR model
    :type var: VAR
    :param label: Index of class to condition the model with
    :type label: int
    :param batch_size: Number of output images
    :type batch_size: int
    :param cfg_w: Classifier guidance rate
    :type cfg_w: float | None
    :param atn_w: Attention guidance rate
    :type atn_w: float | None

    :return: Sampled images
    :rtype: ndarray
    """
    with torch.inference_mode():
        with torch.autocast(DEVICE, enabled=True, dtype=torch.float16, cache_enabled=True):
            ims = var.autoregressive_infer_cfg(
                B=batch_size,
                label_B=label,
                w_cfg=w_cfg,
                w_igg=w_igg,
                top_p=.96,
                top_k=900,
                g_seed=None
            )
        ims = ims.permute(0, 2, 3, 1).mul_(255).numpy()
    return ims


def export(dest: str, ims: np.ndarray, label: str, off: int):
    """
    Export a batch of images.

    :param dest: Path to destination directory
    :type dest: str
    :param ims: Images to export
    :type ims: ndarray
    :param label: Name of the class the images belong to
    :type label: str
    :param off: Offset to apply when naming the output files
    :type off: int
    """
    for i in range(ims.shape[0]):
        im = PImage.fromarray(ims[i].astype(np.uint8))
        im.save(os.path.join(dest, f"{label}_{i + off + 1}.png"))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--depth", type=int, help="VAR depth", choices=DEPTHS)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size")
    parser.add_argument("-c", "--w_cfg", type=float, help="classifier-free guidance scale")
    parser.add_argument("-i", "--w_igg", type=float, help="information-grounding guidance scale")
    args = parser.parse_args()

    depth = args.depth if args.depth else 16
    batch_size = args.batch_size if args.batch_size else 25
    w_cfg: float | None = args.w_cfg
    w_igg: float | None = args.w_igg

    # Create temporary directory for sampled images
    out_dir = f"imagenet/d{depth}_{w_cfg}_{w_igg}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=False)

    # Load model
    var = load_model(depth=depth)
    with open("imagenet/imagenet_classes.json", 'r') as f:
        class_label = json.load(f)

    print(f"{w_cfg = }")
    print(f"{w_igg = }")

    # Sample images
    progbar = tqdm(range(NUM_CLASSES), total=NUM_CLASSES)
    for i in progbar:

        j = 0
        while j < NUM_SAMPLES_PER_CLASS // batch_size:
            ims = sample(var, label=i, batch_size=batch_size, w_cfg=w_cfg, w_igg=w_igg)
            lbl = class_label[i].lower().replace(' ', '_')
            export(out_dir, ims, label=lbl, off=j*batch_size)
            j += 1

        if (last_batch_size := NUM_SAMPLES_PER_CLASS % batch_size) > 0:
            ims = sample(var, label=i, batch_size=last_batch_size, w_cfg=w_cfg, w_igg=w_igg)
            lbl = class_label[i].lower().replace(' ', '_')
            export(out_dir, ims, label=lbl, off=j*batch_size)

    # Generate NPZ file from images
    create_npz_from_sample_folder(out_dir)

    # Remove temporary directory and images
    shutil.rmtree(out_dir, ignore_errors=False)
