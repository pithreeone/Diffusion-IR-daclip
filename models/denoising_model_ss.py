import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class DenoisingModelSS(BaseModel):
    def __init__(self, opt, opt_ff=None):
        super(DenoisingModelSS, self).__init__(opt)
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.amp_dtype = dtype_map[opt['network_G']['amp_dtype']]
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        # if self.rank != -1:
        #     self.device = torch.device(f"cuda:{self.rank}")
        if opt_ff is not None:
            self.opt_ff = opt_ff
            self.fs_model = networks.define_G(opt_ff).to(self.device)
            # self.fs_model.eval()

            # for param in self.fs_model.parameters():
            #     param.requires_grad = False

        if opt["network_G"]["setting"]["cond_type"] == 'concat':
            self.ss_model = networks.define_G(opt, in_ch_scale=3).to(self.device)
        elif opt["network_G"]["setting"]["cond_type"] == 'cond_module':
            self.ss_model = networks.define_G(opt, in_ch_scale=2).to(self.device)
        # self.ss_model = networks.define_G(opt, in_ch_scale=2).to(self.device)
        if opt["dist"]:
            self.ss_model = DistributedDataParallel(
                self.ss_model, device_ids=[torch.cuda.current_device()]
            )
            # self.fs_model = DistributedDataParallel(
            #     self.fs_model, device_ids=[torch.cuda.current_device()]
            # )
        else:
            self.ss_model = DataParallel(self.ss_model)
        # print network
        # self.print_network()
        self.load_fs()
        self.load_ss()

        if self.is_train:
            self.ss_model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.diff_weight = opt['train']['diffusion_weight']
            self.fid_weight = opt['train']['fidelity_weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            fs_optim_params = []
            for (
                k,
                v,
            ) in self.ss_model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            for (k, v) in self.fs_model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    fs_optim_params.append(v)
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
                self.optimizer_fs = torch.optim.Adam(
                    fs_optim_params,
                    lr=train_opt["lr_FS"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1_FS"], train_opt["beta2_FS"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizer_fs = torch.optim.Adam(
                    fs_optim_params,
                    lr=train_opt["lr_FS"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1_FS"], train_opt["beta2_FS"]),
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

            self.ema = EMA(self.ss_model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

            self.crossentropy = nn.CrossEntropyLoss()


    def feed_data(self, state, LQ, GT=None, FS=None, deg_type=None):
        self.state = state.to(self.device)    # noisy_state
        # self.condition = LQ.to(self.device)  # LQ
        self.LQ = LQ.to(self.device)  # LQ
        if FS is not None:
            # self.first_stage_result = FS.to(self.device) # FS
            self.FS = FS.to(self.device) # FS
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT
        if deg_type is not None:
            self.deg_type = deg_type.to(self.device)

    def optimize_parameters(self, step, timesteps, sde=None):
        ### set terminal-state as FS
        self.optimizer.zero_grad()
        # self.optimizer_fs.zero_grad()

        timesteps = timesteps.to(self.device)

        ### First stage prediction
        if self.FS is None:
            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                self.FS = self.fs_model(self.LQ)
            # print(self.LQ.shape, self.FS.shape)



        ### set terminal-state as FS
        # sde.set_mu(self.FS)
        # noise = sde.noise_fn_cond(self.state, self.FS, self.LQ, timesteps.squeeze())
        
        sde.set_mu(self.LQ)
        with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
            noise = sde.noise_fn_cond(self.state, self.LQ, self.FS, timesteps.squeeze())
            # noise, logits = sde.noise_fn_cond(self.state, self.LQ, self.FS, timesteps.squeeze())
        
            score = sde.get_score_from_noise(noise, timesteps)

            # Learning the maximum likelihood objective for state x_{t-1}
            xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
            xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)

            diffusion_loss = self.loss_fn(xt_1_expection, xt_1_optimum)
            # fidelity_loss = self.loss_fn(self.FS, self.state_0)

            # cls_loss = self.crossentropy(logits, self.deg_type)

            loss = self.diff_weight * diffusion_loss
            # loss = self.diff_weight * diffusion_loss + cls_loss
            # loss = self.diff_weight * diffusion_loss + self.fid_weight * fidelity_loss

            loss.backward()
        self.optimizer.step()
        # self.optimizer_fs.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()
        # self.log_dict["diffusion_loss"] = diffusion_loss.item()
        # self.log_dict["fidelity_loss"] = fidelity_loss.item()

    def test(self, sde=None, mode='posterior', save_states=False):
        # sde.set_mu(self.condition)
        self.ss_model.eval()
        with torch.no_grad():
            if mode == 'sde':
                self.output = sde.reverse_sde(self.state, save_states=save_states, text_context=self.text_context, image_context=self.image_context)
            elif mode == 'posterior':
                sde.set_model(self.ss_model)

                ### set terminal-state as FS
                # sde.set_mu(self.FS)
                # self.output = sde.reverse_posterior_cond(self.state, cond=self.LQ, save_states=save_states)

                sde.set_mu(self.LQ)
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    # self.output = sde.reverse_posterior_cond(self.state, cond=self.FS, save_states=save_states)
                    self.output, logits = sde.reverse_posterior_cond(self.state, cond=self.FS, save_states=save_states)

                # pred = logits.argmax(dim=1)   # shape: (N,)
                # print(pred, self.deg_type)
                # self.correct = (pred == self.deg_type).sum().item()
                # self.total = self.deg_type.size(0)

            elif mode == 'posterior_two_stage':
                # First stage prediction
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    self.FS = self.fs_model(self.LQ)

                # Second stage prediction
                sde.set_model(self.ss_model)
                sde.set_mu(self.LQ)
                ss_noisy_state = sde.noise_state(self.LQ)
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    self.output = sde.reverse_posterior_cond(ss_noisy_state, cond=self.FS, save_states=save_states)
                    # self.output, _ = sde.reverse_posterior_cond(ss_noisy_state, cond=self.FS, save_states=save_states)
            elif mode == 'classifier':
                self.FS = self.fs_model(self.LQ)
                timesteps = torch.zeros((self.FS.shape[0]))
                noise, logits = sde.noise_fn_cond(self.state, self.LQ, self.FS, timesteps)
                pred = logits.argmax(dim=1)   # shape: (N,)
                # print(pred, self.deg_type)
                self.correct = (pred == self.deg_type).sum().item()
                self.total = self.deg_type.size(0)
                # print(self.correct, self.total)
                self.output = self.FS

        self.ss_model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        # out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Input"] = self.LQ.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        out_dict["FS"] = self.FS
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load_ss(self):
        load_path_G = self.opt["path"]["ss_pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.ss_model, self.opt["path"]["strict_load"])

    def load_fs(self):
        load_path_G = self.opt_ff["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.fs_model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.ss_model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
        self.save_network(self.fs_model, "FF", iter_label)
