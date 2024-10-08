import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rl_games.algos_torch import torch_ext

from easydict import EasyDict as edict

import os
import time
import sys
sys.path.insert(0, os.getcwd())
import datetime
import logging
import os.path as osp

from tqdm.auto import tqdm
from omegaconf import OmegaConf

from confimo.config import parse_args
from confimo.data.get_data import get_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mlp(loading_keys, checkpoint, actvation_func):
    
    loading_keys_linear = [k for k in loading_keys if k.endswith('weight')]
    nn_modules = []
    for idx, key in enumerate(loading_keys_linear):
        if len(checkpoint['model'][key].shape) == 1: # layernorm
            layer = torch.nn.LayerNorm(*checkpoint['model'][key].shape[::-1])
            nn_modules.append(layer)
        elif len(checkpoint['model'][key].shape) == 2: # nn
            layer = nn.Linear(*checkpoint['model'][key].shape[::-1])
            nn_modules.append(layer)
            if idx < len(loading_keys_linear) - 1:
                nn_modules.append(actvation_func())
        else:
            raise NotImplementedError
        
    net = nn.Sequential(*nn_modules)
    
    state_dict = net.state_dict()
    
    for idx, key_affix in enumerate(state_dict.keys()):
        state_dict[key_affix].copy_(checkpoint['model'][loading_keys[idx]])
        
    for param in net.parameters():
        param.requires_grad = False
        
    return net


def load_linear(net_name, checkpoint):
    net = nn.Linear(checkpoint['model'][net_name + '.weight'].shape[1], checkpoint['model'][net_name + '.weight'].shape[0])
    state_dict = net.state_dict()
    state_dict['weight'].copy_(checkpoint['model'][net_name + '.weight'])
    state_dict['bias'].copy_(checkpoint['model'][net_name + '.bias'])
    
    return net


def load_encoder(checkpoint, device):
    encoder = edict()

    net_key_name = "a2c_network._task_mlp" if "a2c_network._task_mlp.0.weight" in checkpoint['model'].keys() else "a2c_network.z_mlp"

    loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(net_key_name)]
    actor = load_mlp(loading_keys, checkpoint, nn.SiLU)

    actor.to(device)
    actor.eval()

    encoder.encoder = actor
    if "a2c_network.z_logvar.weight" in checkpoint['model'].keys():
        z_logvar = load_linear('a2c_network.z_logvar', checkpoint=checkpoint)
        z_mu = load_linear('a2c_network.z_mu', checkpoint=checkpoint)
        z_logvar.eval()
        z_mu.eval()
        encoder.z_mu = z_mu.to(device)
        encoder.z_logvar = z_logvar.to(device)

    return encoder


def load_decoder(checkpoint, device):
    key_name = "a2c_network.actor_mlp"
    loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(key_name)] + ["a2c_network.mu.weight", 'a2c_network.mu.bias']

    actor = load_mlp(loading_keys, checkpoint, nn.SiLU)

    actor.to(device)
    actor.eval()

    decoder = edict()
    decoder.decoder= actor

    prior_loading_keys = [k for k in checkpoint['model'].keys() if k.startswith("a2c_network.z_prior.")]
    z_prior = load_mlp(prior_loading_keys, checkpoint, nn.SiLU)
    z_prior.append(nn.SiLU())
    z_prior_mu = load_linear('a2c_network.z_prior_mu', checkpoint=checkpoint)
    z_prior.eval()
    z_prior_mu.eval()

    decoder.z_prior = z_prior.to(device)
    decoder.z_prior_mu = z_prior_mu.to(device)

    if "a2c_network.z_prior_logvar.weight" in checkpoint['model'].keys():
        z_prior_logvar = load_linear('a2c_network.z_prior_logvar', checkpoint=checkpoint)
        z_prior_logvar.eval()
        decoder.z_prior_logvar = z_prior_logvar.to(device)

    return decoder


def load_encoder_decoder():

    ######## dimension of PULSE ########
    # s^p: 358    s^g: 576    z: 32    a: 69

    ## encoder
    # encoder:                   934 --> 160
    # z_mu/z_logvar:             160 --> 32

    ## decoder
    # decoder:                   390 --> 69
    # z_prior:                   358 --> 512
    # z_prior_mu/z_prior_logvar: 512 --> 32

    cfg = parse_args()
    if cfg.vis_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.vis_gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    set_seed(cfg.TEST.SEED_VALUE)

    name_time_str = osp.join(cfg.NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    output_dir = osp.join(cfg.FOLDER, name_time_str)
    os.makedirs(output_dir, exist_ok=False)

    steam_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(output_dir, 'output.log'))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[steam_handler, file_handler])
    logger = logging.getLogger(__name__)

    OmegaConf.save(cfg, osp.join(output_dir, 'config.yaml'))

    logger.info(f"device: {device}")

    checkpoint = torch_ext.load_checkpoint(cfg.model.checkpoint_path)
    encoder = load_encoder(checkpoint, device)
    decoder = load_decoder(checkpoint, device)

    logger.info(f"encoder:\n{encoder}\ndecoder:\n{decoder}")


if __name__ == "__main__":
    load_encoder_decoder()