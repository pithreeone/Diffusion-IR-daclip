import logging
from collections import OrderedDict

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from ema_pytorch import EMA

import models.networks as networks
# from models.modules.UNet_arch import ConditionalUNet
from models.modules.loss import MatchingLoss
from models.util import instantiate_from_config

from .base_model import BaseModel

logger = logging.getLogger("base")

class FeedForwardModel(BaseModel):
    def __init__(self, opt):
        super(FeedForwardModel, self).__init__(opt)
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.amp_dtype = dtype_map[opt['network_G']['amp_dtype']]

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt["train"]

        if self.rank != -1:
            self.device = torch.device(f"cuda:{self.rank}")
        
        self.model = networks.define_G(opt).to(self.device)
        # opt["dist"] = False
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.model = DataParallel(self.model)
        
        self.load()

        # self.loss: torch.nn.Module = instantiate_from_config(opt["train"]["loss_config"]).to(self.device)

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, input, target=None):
        # print(self.device)
        self.input = input.to(self.device)    # noisy_state
        # self.condition = LQ.to(self.device)  # LQ
        if target is not None:
            self.target = target.to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
            output = self.model(self.input)
        # output = self.model(self.target)

        # optimizer_idx = 0
        # if hasattr(self.loss, "forward_keys"):
        #     extra_info = {
        #         # "z": z,
        #         "optimizer_idx": optimizer_idx,
        #         "global_step": step,
        #         "last_layer": self.get_last_layer(),
        #         "split": "train",
        #         # "regularization_log": regularization_log,
        #         "autoencoder": self,
        #     }
        #     extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        # else:
        #     extra_info = dict()

        loss = self.loss_fn(output, self.target)
        # loss, log = self.loss(self.target, output, **extra_info)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        self.log_dict["loss"] = loss.item()

        # merge other metrics from log dict
        # self.log_dict.update(log)

    def test(self):

        # print(self.device)
        self.input = self.input.to(next(self.model.parameters()).device)
        self.model.eval()
        # print(next(self.model.parameters()).device)
        # print(self.input.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                self.output = self.model(self.input)
            # self.output = self.model(self.target)
        
        self.model.train()

        return self.output

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def get_last_layer(self):
        return self.model.module.get_last_layer()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.input.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.target.detach()[0].float().cpu()
        return out_dict

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')