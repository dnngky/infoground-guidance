# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
# pylint: disable=unspecified-encoding

"""
Evaluation on attention-based guidance.
"""
import json
import os
import os.path as osp

import numpy as np
import PIL.Image as PImage
import torch
from tqdm import tqdm

from utils.misc import create_npz_from_sample_folder
from models import build_vae_var, VAR

# disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda _: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda _: None)

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

NUM_CLASSES = 1000
NUM_SAMPLES_PER_CLASS = 50


def load_models(depth: int) -> VAR:
    """
    Load a VAR model with given depth.
    
    :param depth: Depth of VAR model to be loaded
    :type depth: int
    
    :return: Loaded VAR model
    :rtype: VAR
    """
    if depth not in (16, 20, 24, 30, 36):
        raise ValueError("Invalid depth")

    # download checkpoint
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt = f"state_dicts/vae_ch160v4096z32_d{depth}.pth"
    var_ckpt = f"state_dicts/var_d{depth}.pth"

    if not osp.exists(vae_ckpt):
        vae_ckpt_old = vae_ckpt.replace(f"_d{depth}", "")
        os.system(f'wget {hf_home}/{vae_ckpt_old}')
        os.rename(vae_ckpt_old, vae_ckpt)
    if not osp.exists(var_ckpt):
        os.system(f'wget {hf_home}/{var_ckpt}')

    # build vae and var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=NUM_CLASSES, depth=depth, shared_aln=False,
    )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
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
    cfg: int | float | list[torch.Tensor],
    use_attn: bool = False
) -> np.ndarray:
    """
    Sample a batch of images from given VAR model.
    
    :param var: VAR model
    :type var: VAR
    :param label: Index of class to condition the model with
    :type label: int
    :param batch_size: Number of output images
    :type batch_size: int
    :param cfg: Classifier guidance rate
    :type cfg: float
    :param use_attn: Whether to use attention-based guidance
    :type use_attn: bool
    
    :return: Sampled images
    :rtype: ndarray
    """
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
            ims = var.autoregressive_infer_cfg(
                B=batch_size,
                label_B=label,
                cfg=cfg,
                top_p=.96,
                top_k=900,
                use_attn=use_attn,
                g_seed=None
            )
        ims = ims.permute(0, 2, 3, 1).mul_(255).cpu().numpy()
    return ims


def export(dest: str, label: str, off: int, ims: np.ndarray):
    """
    Export a batch of images.
    
    :param dest: Path to destination directory
    :type dest: str
    :param label: Name of the class the images belong to
    :type label: str
    :param off: Offset to apply when naming the output files
    :type off: int
    :param ims: Images to export
    :type ims: ndarray
    """
    for i in range(ims.shape[0]):
        im = PImage.fromarray(ims[i].astype(np.uint8))
        im.save(os.path.join(dest, f"{label}_{i + off + 1}.png"))


if __name__ == "__main__":

    DEPTH = 16
    BATCH_SIZE = 25
    USE_ATTN = False

    var = load_models(depth=DEPTH)
    with open("imagenet_in/imagenet_classes.json", 'r') as f:
        class_label = json.load(f)

    progbar = tqdm(
        range(156, NUM_CLASSES),
        total=NUM_CLASSES - 156,
        leave=False
    )
    out_dir = f"imagenet_out/d{DEPTH}_{'attn' if USE_ATTN else 'vanilla'}"

    for i in progbar:
        progbar.set_description(f"[{i}] {class_label[i]}")
        for j in range(NUM_SAMPLES_PER_CLASS // BATCH_SIZE):
            ims = sample(var, label=i, batch_size=BATCH_SIZE, cfg=1.5, use_attn=USE_ATTN)
            lbl = class_label[i].lower().replace(' ', '_')
            export(out_dir, lbl, j * BATCH_SIZE, ims)

    create_npz_from_sample_folder(out_dir)
