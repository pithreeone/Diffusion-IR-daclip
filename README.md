## How to Run
### Configuration
Most parameters are specified in the `options/` folder. The following list shows the common parameters used in each file:
- `distortion`: Specify the degradation type.  
  *Choices: `motion-blurry`, `hazy`, `low-light`, `noisy`, `rainy`*

- `gpu_ids`: List of GPU IDs to use.  
  *Example: `[0]` for single GPU, `[0,1]` for two GPUs.*

### Feed-forward model
#### Training
- multiple GPU:
  `torchrun --nproc_per_node=2 --master_port=4321 train_ff.py --opt=options/train_ff.yaml --launcher pytorch`

Modify the hyperparameters in `options/train_ff.yaml`:
- `datasets.train.batch_size`: total batch-size (not per GPU).
- `network_G.setting.nf`: number of channels in the first stage.  

#### Testing
`python3 test_ff.py --opt options/test_ff.yaml`

Modify the model path in `options/test_ff.yaml`.
- `path.pretrain_model_G`: path to the pretrained model

---
### Diffusion-IR:
#### Train
- single GPU: 
 `python3 train.py --opt options/train.yaml`

- multiple GPU:
 `torchrun --nproc_per_node=2 --master_port=4321 train.py --opt=options/train.yaml --launcher pytorch`


#### Configuration:
Modify parameters in `options/train.yaml`

- `datasets.train.dataroot`: Path to the training dataset.  
  *(Include sub-datasets for each degradation type.)*

- `datasets.val.dataroot`: Path to the validation dataset.  
  *(Use the validation set provided in DA-CLIP.)*

#### Test:
Test model without da-clip embedding:
`python3 test.py --opt options/test.yaml`

Test model with da-clip embedding:
`python3 test.py --opt options/test_daclip.yaml`

#### Configuration:
Modify parameters in `options/test.yaml`
- `gpu_ids`: List of GPU IDs to use.  
  *(Currently supports only a single GPU.)*

- `path.pretrain_model_G`: Path to the pretrained generator model parameters.

- `datasets.test.dataroot_GT`, `datasets.test.dataroot_LQ`:  
  Paths to the HQ (ground truth) and LQ (degraded) testing datasets.

Modify parameters in `options/test_daclip.yaml`
- `gpu_ids`: List of GPU IDs to use.  
  *(Currently supports only a single GPU.)*

- `path.pretrain_model_G`: Path to the pretrained generator model parameters.

- `path.daclip`: Path to the pretrained daclip model parameters.

- `datasets.test.dataroot_GT`, `datasets.test.dataroot_LQ`:  
  Paths to the HQ (ground truth) and LQ (degraded) testing datasets.

---

### Two-stage Training
After training the first-stage feed-forward IR model, we use its predictions as a conditioning input to train the second-stage diffusion-based IR model.
#### Train
- single GPU: 
 `python3 train_cascade_ff.py --opt options/train_cascade_ff.yaml`

- multiple GPU:
 `torchrun --nproc_per_node=2 --master_port=4321 train_cascade_ff.py --opt=options/train_cascade_ff.yaml --launcher pytorch`

#### Configuration:
Set the path to the pretrained first-stage model in `options/ff.yaml`:
- `path.pretrain_model_G`: Path to the pretrained first stage (feed-forward) model.


Modify training parameters in `options/train_cascade_ff.yaml`
- `gpu_ids`: List of GPU IDs to use.

- `datasets.train.batch_size`: Samples per batch for training.

- `datasets.train.dataroot`: Path to the training dataset.  
  *(Include sub-datasets for each degradation type.)*

- `datasets.val.dataroot`: Path to the validation dataset.  
  *(Use the validation set provided in DA-CLIP.)*

- `network_G.setting.use_deg_embedding`: Whethere to use degradation embedding.

- `network_G.setting.attn`: The attention type in UNet block.

- `network_G.amp_dtype`: Set the model’s precision. Use `bfloat16` to accelerate training and reduce memory usage, or `float32` for full precision.