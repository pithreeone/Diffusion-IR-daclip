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
    def __init__(self, opt):
        super(DenoisingModelSS, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.fs_model = networks.define_G(opt, in_ch_scale=2).to(self.device)
        self.fs_model.eval()
        for param in self.fs_model.parameters():
            param.requires_grad = False
        if opt["network_G"]["setting"]["cond_type"] == 'concat':
            self.ss_model = networks.define_G(opt, in_ch_scale=3).to(self.device)
        elif opt["network_G"]["setting"]["cond_type"] == 'cond_module':
            self.ss_model = networks.define_G(opt, in_ch_scale=2).to(self.device)
        # self.ss_model = networks.define_G(opt, in_ch_scale=2).to(self.device)
        if opt["dist"]:
            self.ss_model = DistributedDataParallel(
                self.ss_model, device_ids=[torch.cuda.current_device()]
            )
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
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.ss_model.named_parameters():  # can optimize for a part of the model
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

            self.ema = EMA(self.ss_model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, state, LQ, GT=None, FS=None, text_context=None, image_context=None):
        self.state = state.to(self.device)    # noisy_state
        # self.condition = LQ.to(self.device)  # LQ
        self.LQ = LQ.to(self.device)  # LQ
        if FS is not None:
            # self.first_stage_result = FS.to(self.device) # FS
            self.FS = FS.to(self.device) # FS
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT
        self.text_context = text_context
        self.image_context = image_context

    def optimize_parameters(self, step, timesteps, sde=None):
        ### set terminal-state as FS
        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)

        # Get noise and score
        ### set terminal-state as FS
        sde.set_mu(self.FS)
        noise = sde.noise_fn_cond(self.state, self.FS, self.LQ, timesteps.squeeze())
        # sde.set_mu(self.LQ)
        # noise = sde.noise_fn_cond(self.state, self.LQ, self.FS, timesteps.squeeze())
        
        score = sde.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def test(self, sde=None, mode='posterior', save_states=False):
        # sde.set_mu(self.condition)
        self.ss_model.eval()
        with torch.no_grad():
            if mode == 'sde':
                self.output = sde.reverse_sde(self.state, save_states=save_states, text_context=self.text_context, image_context=self.image_context)
            elif mode == 'posterior':
                sde.set_model(self.ss_model)

                ### set terminal-state as FS
                sde.set_mu(self.FS)
                self.output = sde.reverse_posterior_cond(self.state, cond=self.LQ, save_states=save_states)

                # sde.set_mu(self.LQ)
                # self.output = sde.reverse_posterior_cond(self.state, cond=self.FS, save_states=save_states)
                
            elif mode == 'posterior_test':
                # First stage prediction
                sde.set_model(self.fs_model)
                sde.set_mu(self.condition)
                self.fs_output = sde.reverse_posterior(self.state, save_states=save_states, text_context=self.text_context, image_context=self.image_context)
                
                # Second stage prediction
                sde.set_model(self.ss_model)
                sde.set_mu(self.fs_output)
                ss_noisy_state = sde.noise_state(self.fs_output)
                self.output = sde.reverse_posterior_cond(ss_noisy_state, cond=self.condition, save_states=save_states, text_context=self.text_context, image_context=self.image_context)

        self.ss_model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        # out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Input"] = self.LQ.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
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
        load_path_G = self.opt["path"]["fs_pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.fs_model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.ss_model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
