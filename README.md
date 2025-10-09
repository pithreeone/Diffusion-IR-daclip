## How to Run
### Train (Feed-forward model)
- multiple GPU:
  `torchrun --nproc_per_node=2 --master_port=4321 train_ff.py --opt=options/train_ff.yaml --launcher pytorch`


### Train (Diffusion-IR):

- single GPU: 
 `python3 train.py --opt options/train.yaml`

- multiple GPU:
 `python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 train.py -opt=options/train.yaml --launcher pytorch`


#### Configuration:
Modify parameters in `options/train.yaml`

- **`distortion`**: Specify the degradation type.  
  *Choices: `motion-blurry`, `hazy`, `low-light`, `noisy`, `rainy`*

- **`gpu_ids`**: List of GPU IDs to use.  
  *Example: `[0]` for single GPU, `[0,1]` for two GPUs.*

- **`datasets.train.dataroot`**: Path to the training dataset.  
  *(Include sub-datasets for each degradation type.)*

- **`datasets.val.dataroot`**: Path to the validation dataset.  
  *(Use the validation set provided in DA-CLIP.)*

### Test:
Test model without da-clip embedding:
`python3 test.py --opt options/test.yaml`

Test model with da-clip embedding:
`python3 test.py --opt options/test_daclip.yaml`

#### Configuration:
Modify parameters in `options/test.yaml`
- **`gpu_ids`**: List of GPU IDs to use.  
  *(Currently supports only a single GPU.)*

- **`path.pretrain_model_G`**: Path to the pretrained generator model parameters.

- **`datasets.test.dataroot_GT`**, **`datasets.test.dataroot_LQ`**:  
  Paths to the HQ (ground truth) and LQ (degraded) testing datasets.

Modify parameters in `options/test_daclip.yaml`
- **`gpu_ids`**: List of GPU IDs to use.  
  *(Currently supports only a single GPU.)*

- **`path.pretrain_model_G`**: Path to the pretrained generator model parameters.

- **`path.daclip`**: Path to the pretrained daclip model parameters.

- **`datasets.test.dataroot_GT`**, **`datasets.test.dataroot_LQ`**:  
  Paths to the HQ (ground truth) and LQ (degraded) testing datasets.