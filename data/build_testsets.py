import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import json
import os
import yaml
from glob import glob
import random
import argparse
# from utils import dict2namespace, load_img

DEG_MAP = {
    "noise": 0,
    "blur" : 1,
    "rain" : 2,
    "haze" : 3,
    "lol"  : 4,
    "sr"   : 5,
    "en"   : 6,
}

DEG2TASK = {
    "noise": "denoising",
    "blur" : "deblurring",
    "rain" : "deraining",
    "haze" : "dehazing",
    "lol"  : "lol",
    "sr"   : "sr",
    "en"   : "enhancement"
}

def augment_prompt (prompt):
    ### special prompts
    lol_prompts = ["fix the illumination", "increase the exposure of the photo", "the image is too dark to see anything, correct the photo", "poor illumination, improve the shot", "brighten dark regions", "make it HDR", "improve the light of the image", "Can you make the image brighter?"]
    sr_prompts  = ["I need to enhance the size and quality of this image.", "My photo is lacking size and clarity; can you improve it?", "I'd appreciate it if you could upscale this photo.", "My picture is too little, enlarge it.", "upsample this image", "increase the resolution of this photo", "increase the number of pixels", "upsample this photo", "Add details to this image", "improve the quality of this photo"]
    en_prompts  = ["make my image look like DSLR", "improve the colors of my image", "improve the contrast of this photo", "apply tonemapping", "enhance the colors of the image", "retouch the photo like a photograper"]
    
    init = np.random.choice(["Remove the", "Reduce the", "Clean the", "Fix the", "Remove", "Improve the", "Correct the",])
    end  = np.random.choice(["please", "fast", "now", "in the photo", "in the picture", "in the image", ""])
    newp = f"{init} {prompt} {end}"
    
    if "lol" in prompt:
        newp = np.random.choice(lol_prompts)
    elif "sr" in prompt:
        newp = np.random.choice(sr_prompts)
    elif "en" in prompt:
        newp = np.random.choice(en_prompts)

    newp = newp.strip().replace("  ", " ").replace("\n", "")
    return newp

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

def get_deg_name(path):
    """
    Get the degradation name from the path
    """

    if ("gopro" in path) or ("GoPro" in path) or ("blur" in path) or ("Blur" in path) or ("RealBlur" in path):
        return "blur"
    elif ("SOTS" in path) or ("haze" in path) or ("sots" in path) or ("RESIDE" in path): 
        return "haze"
    elif ("LOL" in path):
        return "lol"
    elif ("fiveK" in path):
        return "en"
    elif ("super" in path) or ("classicalSR" in path):
        return "sr"
    elif ("Rain100" in path) or ("rain13k" in path) or ("Rain13k" in path):
        return "rain"
    else:
        return "noise"
    
def crop_img(image, base=16):
    """
    Mod crop the image to ensure the dimension is divisible by base. Also done by SwinIR, Restormer and others.
    """
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


################# DATASETS

import torch
class RefDegImage(Dataset):
    """
    Dataset for Image Restoration having low-quality image and the reference image.
    Tasks: synthetic denoising, deblurring, super-res, etc.
    """
    
    def __init__(self, hq_img_paths, lq_img_paths, process_mode, augmentations=None, name="test", deg_name="noise", deg_class=0):

        assert len(hq_img_paths) == len(lq_img_paths)
        
        self.hq_paths     = hq_img_paths
        self.lq_paths     = lq_img_paths
        self.totensor     = torchvision.transforms.ToTensor()
        # self.val       = val
        self.process_mode = process_mode
        self.augs         = augmentations
        self.name         = name
        self.degradation  = deg_name
        self.deg_class    = deg_class
        self.patch_size   = 256

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.kernel = self.gaussian_kernel(3)

        # if self.val:
        if self.process_mode == 'val':
            self.augs = None # No augmentations during validation/test

    def __len__(self):
        return len(self.hq_paths)

    def __getitem__(self, idx):
        hq_path = self.hq_paths[idx]
        lq_path = self.lq_paths[idx]
        
        hq_image = load_img(hq_path)
        lq_image = load_img(lq_path)

        if self.process_mode == 'val':
            random.seed(42+idx)
            # if an image has an odd number dimension we trim for example from [321, 189] to [320, 188].
            # hq_image = crop_img(hq_image)
            # lq_image = crop_img(lq_image)
        
            H, W, C = hq_image.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            hq_image = hq_image[rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :]
            lq_image = lq_image[rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :]
        elif self.process_mode == 'test':
            size_divisor = 32
            hq_image = crop_img(hq_image, size_divisor)
            lq_image = crop_img(lq_image, size_divisor)

        hq_image = self.totensor(hq_image.astype(np.float32))
        lq_image = self.totensor(lq_image.astype(np.float32))
# 
        # lq_image = (lq_image - lq_image.mean()) / lq_image.std() * 0.2 + 0.45
        # lq_image = F.conv2d(lq_image.unsqueeze(0), self.kernel, padding=2, groups=3).squeeze(0)
        # lq_image = torch.clamp(lq_image, 0, 1)

        # hq_image = hq_image * 2.0 - 1.0
        # lq_image = lq_image * 2.0 - 1.0

        # print(hq_image.shape)
        # return hq_image, lq_image, hq_path
        # return {"GT": hq_image, "LQ": lq_image, "GT_path": hq_path}
        return {
            "GT": hq_image, 
            "LQ": lq_image,
            "GT_path": hq_path, 
            "txt": "", 
            'original_size_as_tuple': torch.tensor([256, 256]),
            'target_size_as_tuple': torch.tensor([256, 256]),
            'crop_coords_top_left': torch.tensor([0, 0]),
        }


    def gaussian_kernel(self, channels, kernel_size=5, sigma=1.0):
        # 1D Gaussian
        x = torch.arange(kernel_size) - kernel_size // 2
        x = torch.exp(-(x**2)/(2*sigma**2))
        x = x / x.sum()
        # 2D Gaussian kernel
        kernel2d = x[:, None] @ x[None, :]
        kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size)
        return kernel2d


def create_testsets (testsets, process_mode='test', debug=False):
    """
    Given a list of testsets create pytorch datasets for each.
    The method requires the paths to references and noisy images.
    """
    assert len(testsets) > 0

    if debug:
        print (20*'****')
        print ("Creating Testsets", len(testsets))
    
    datasets = []
    for testdt in testsets:

        path_hq , path_lq = testdt[0], testdt[1]
        if debug: print (path_hq , path_lq)

        if ("denoising" in path_hq) or ("jpeg" in path_hq):
            dataset_name  = path_hq.split("/")[-1]
            dataset_sigma = path_lq.split("/")[-1].split("_")[-1].split(".")[0]
            dataset_name  = dataset_name+ f"_{dataset_sigma}"
        elif "Rain" in path_hq:
            if "Rain100L" in path_hq:
                dataset_name  = "Rain100L"
            else:
                dataset_name  = path_hq.split("/")[3]

        elif ("gopro" in path_hq) or ("GoPro" in path_hq):
            dataset_name  = "GoPro"
        elif "LOL" in path_hq:
            dataset_name  = "LOL"
        elif "SOTS" in path_hq:
            dataset_name  = "SOTS"
        elif "fiveK" in path_hq:
            dataset_name  = "MIT5K"
        else:
            assert False, f"{path_hq} - unknown dataset"

        hq_img_paths = sorted(glob(os.path.join(path_hq, "*")))
        lq_img_paths = sorted(glob(os.path.join(path_lq, "*")))

        if "SOTS" in path_hq:
            # Haze removal SOTS test dataset
            dataset_name  = "SOTS"
            hq_img_paths = sorted(glob(os.path.join(path_hq, "*.jpg")))
            assert len(hq_img_paths) == 500

            lq_img_paths = [file.replace("GT", "IN") for file in hq_img_paths]

        if "fiveK" in path_hq:
            dataset_name  = "MIT5K"
            testf = "test-data/mit5k/test.txt"
            f = open(testf, "r")
            test_ids = f.readlines()
            test_ids = [x.strip() for x in test_ids]
            f.close()
            hq_img_paths = [os.path.join(path_hq, f"{x}.jpg") for x in test_ids]
            lq_img_paths = [x.replace("expertC", "input") for x in hq_img_paths]
            assert len(hq_img_paths) == 498
        
        if "gopro" in path_hq:
            assert len(hq_img_paths) == 1111

        if "LOL" in path_hq:
            assert len(hq_img_paths) == 15

        assert len(hq_img_paths) == len(lq_img_paths)

        deg_name  = get_deg_name(path_hq)
        deg_class = DEG_MAP[deg_name]

        valdts = RefDegImage(hq_img_paths = hq_img_paths,
                            lq_img_paths  = lq_img_paths,
                            process_mode = process_mode, name= dataset_name, deg_name=deg_name, deg_class=deg_class)
        
        datasets.append(valdts)
    
    assert len(datasets) == len(testsets)
    print (20*'****')

    return datasets

def get_testsets(config, process_mode='test'):
    # parse config file
    with open(os.path.join(config), "r") as f:
        config = yaml.safe_load(f)

    cfg = dict2namespace(config)

    ################### TESTING DATASET

    TESTSETS      = []
    dn_testsets   = []
    rain_testsets = []

    # Denoising
    try:
        for testset in cfg.test.dn_datasets:
            for sigma in cfg.test.dn_sigmas:
                noisy_testpath = os.path.join(cfg.test.dataset_root, cfg.test.dn_datapath, testset+ f"_{sigma}")
                clean_testpath = os.path.join(cfg.test.dataset_root, cfg.test.dn_datapath, testset)
                #print (clean_testpath, noisy_testpath)
                dn_testsets.append([clean_testpath, noisy_testpath])
    except:
        dn_testsets = []

    # RAIN
    try:
        for noisy_testpath, clean_testpath in zip(cfg.test.rain_inputs, cfg.test.rain_targets):
            rain_testsets.append([os.path.join(cfg.test.dataset_root, clean_testpath), 
                                  os.path.join(cfg.test.dataset_root, noisy_testpath)])
    except:
        rain_testsets = []

    # HAZE
    try:
        haze_testsets = [[os.path.join(cfg.test.dataset_root, cfg.test.haze_targets), 
                          os.path.join(cfg.test.dataset_root, cfg.test.haze_inputs)]]
    except:
        haze_testsets = []

    # BLUR
    try:
        blur_testsets = [[os.path.join(cfg.test.dataset_root, cfg.test.gopro_targets), 
                          os.path.join(cfg.test.dataset_root, cfg.test.gopro_inputs)]]
    except:
        blur_testsets = []

    # LOL
    try:
        lol_testsets = [[os.path.join(cfg.test.dataset_root, cfg.test.lol_targets), 
                         os.path.join(cfg.test.dataset_root, cfg.test.lol_inputs)]]
    except:
        lol_testsets = []

    # MIT5K
    try:
        mit_testsets = [[os.path.join(cfg.test.dataset_root, cfg.test.mit_targets), os.path.join(cfg.dataset_root, cfg.test.mit_inputs)]]
    except:
        mit_testsets = []

    TESTSETS += dn_testsets
    TESTSETS += rain_testsets
    TESTSETS += haze_testsets
    TESTSETS += blur_testsets
    TESTSETS += lol_testsets
    TESTSETS += mit_testsets
    
    # print ("Tests:", TESTSETS)
    print ("TOTAL TESTSET:", len(TESTSETS))
    print (20 * "----")
    # print(TESTSETS)
    for sets in TESTSETS:
        print(sets)

    test_datasets = create_testsets(TESTSETS, process_mode=process_mode, debug=True)

    return test_datasets