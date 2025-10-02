import torch
import numpy as np
import random
import cv2
from PIL import Image
import argparse
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from sgm.util import instantiate_from_config
# from scripts.demo.streamlit_helpers import load_model_from_config

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True 

def save_rgb (img, filename):
    
    img = np.clip(img, 0., 1.)
    if np.max(img) <= 1:
        img = img * 255
    
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def load_img (filename, norm=True,):
    img = np.array(Image.open(filename).convert("RGB"))
    if norm:   
        img = img / 255.
        img = img.astype(np.float32)
    return img

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def initial_model_load(model, lowvram_mode=False, device='cuda'):
    if lowvram_mode:
        model.model.half()
    else:
        model.to(device)
    return model

def load_model_from_config(config, ckpt=None, verbose=True, **kwargs):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            print(pl_sd.keys())
            if "global_step" in pl_sd:
                global_step = pl_sd["global_step"]
                print(f"loaded ckpt from global step {global_step}")
                print(f"Global Step: {pl_sd['global_step']}")
            sd = pl_sd["state_dict"]

        elif ckpt.endswith("safetensors"):
            sd = load_safetensors(ckpt)
        else:
            raise NotImplementedError

        msg = None

        m, u = model.load_state_dict(sd, strict=False)

        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        msg = None

    # model = initial_model_load(model, **kwargs)
    model.eval()
    return model, msg

def get_model(model_config, model_path=None, **kwargs):
    config = OmegaConf.load(model_config)
    model, msg = load_model_from_config(config, model_path, **kwargs)
    return model