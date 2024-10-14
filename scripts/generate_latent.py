import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rl_games.algos_torch import torch_ext

from easydict import EasyDict as edict

import os
from os.path import join
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

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from scipy.spatial.transform import Rotation as sRot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
import joblib

import pandas as pd

from phc.env.tasks.humanoid_z import HumanoidZ

upright_start = True
robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, 
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False, 
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smpl",
    }
smpl_local_robot = LocalRobot(robot_cfg,)


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
    return encoder, decoder


def store_motion(src_path, mirror=True):

    entry_data = dict(np.load(open(src_path, "rb"), allow_pickle=True))

    framerate = entry_data['mocap_framerate']
    skip = int(framerate/30)

    root_trans = entry_data['trans'][::skip, :]
    pose_aa = np.concatenate([entry_data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]

    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

    beta = np.zeros((16))
    gender_number, beta[:], gender = [0], 0, "neutral"

    smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
    smpl_local_robot.write_xml(f"data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    skeleton_tree = SkeletonTree.from_mjcf(f"data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
    
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                torch.from_numpy(pose_quat),
                root_trans_offset,
                is_local=True)
    
    if robot_cfg['upright_start']:
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
        pose_quat = new_sk_state.local_rotation.numpy()
    
    pose_quat_global = new_sk_state.global_rotation.numpy()
    pose_quat = new_sk_state.local_rotation.numpy()
    fps = 30

    new_motion_out = {}
    new_motion_out['pose_quat_global'] = pose_quat_global
    new_motion_out['pose_quat'] = pose_quat
    new_motion_out['trans_orig'] = root_trans
    new_motion_out['root_trans_offset'] = root_trans_offset
    new_motion_out['beta'] = beta
    new_motion_out['gender'] = gender
    new_motion_out['pose_aa'] = pose_aa
    new_motion_out['fps'] = fps

    if mirror:

        left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
        
        pose_quat_global = pose_quat_global[:, left_to_right_index]
        pose_quat_global[..., 0] *= -1
        pose_quat_global[..., 2] *= -1

        pose_quat = pose_quat[:, left_to_right_index]
        pose_quat[..., 0] *= -1
        pose_quat[..., 2] *= -1

        root_trans[..., 1] *= -1
        root_trans_offset[..., 1] *= -1

        pose_aa = pose_aa[:, left_to_right_index]
        pose_aa[..., 1] *= -1

        M_new_motion_out = {}
        M_new_motion_out['pose_quat_global'] = pose_quat_global
        M_new_motion_out['pose_quat'] = pose_quat
        M_new_motion_out['trans_orig'] = root_trans
        M_new_motion_out['root_trans_offset'] = root_trans_offset
        M_new_motion_out['beta'] = beta
        M_new_motion_out['gender'] = gender
        M_new_motion_out['pose_aa'] = pose_aa
        M_new_motion_out['fps'] = fps
    
    return new_motion_out, M_new_motion_out


def store_motion_humanact12(src_path, mirror=True):

    entry_data = joblib.load(src_path)

    skip = 1

    # trans_matrix = np.array(
    #     [[0, 1, 0], 
    #     [0, 0, -1], 
    #     [1, 0, 0]]
    # )
    trans_matrix = np.array(
        [[1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1]]
    )
    root_trans = np.dot(entry_data['cam'][::skip, :], trans_matrix)
    pose_aa = np.concatenate([entry_data['pose'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['beta']
    N = pose_aa.shape[0]

    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

    beta = np.zeros((16))
    gender_number, beta[:], gender = [0], 0, "neutral"

    smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
    smpl_local_robot.write_xml(f"data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    skeleton_tree = SkeletonTree.from_mjcf(f"data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
    
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                torch.from_numpy(pose_quat),
                root_trans_offset,
                is_local=True)
    
    if robot_cfg['upright_start']:
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
        pose_quat = new_sk_state.local_rotation.numpy()
    
    pose_quat_global = new_sk_state.global_rotation.numpy()
    pose_quat = new_sk_state.local_rotation.numpy()
    fps = 30    # Here it actually 20 fps, we ignore it for avoiding potential puzzles further.

    new_motion_out = {}
    new_motion_out['pose_quat_global'] = pose_quat_global
    new_motion_out['pose_quat'] = pose_quat
    new_motion_out['trans_orig'] = root_trans
    new_motion_out['root_trans_offset'] = root_trans_offset
    new_motion_out['beta'] = beta
    new_motion_out['gender'] = gender
    new_motion_out['pose_aa'] = pose_aa
    new_motion_out['fps'] = fps

    if mirror:

        left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
        
        pose_quat_global = pose_quat_global[:, left_to_right_index]
        pose_quat_global[..., 0] *= -1
        pose_quat_global[..., 2] *= -1

        pose_quat = pose_quat[:, left_to_right_index]
        pose_quat[..., 0] *= -1
        pose_quat[..., 2] *= -1

        root_trans[..., 1] *= -1
        root_trans_offset[..., 1] *= -1

        pose_aa = pose_aa[:, left_to_right_index]
        pose_aa[..., 1] *= -1

        M_new_motion_out = {}
        M_new_motion_out['pose_quat_global'] = pose_quat_global
        M_new_motion_out['pose_quat'] = pose_quat
        M_new_motion_out['trans_orig'] = root_trans
        M_new_motion_out['root_trans_offset'] = root_trans_offset
        M_new_motion_out['beta'] = beta
        M_new_motion_out['gender'] = gender
        M_new_motion_out['pose_aa'] = pose_aa
        M_new_motion_out['fps'] = fps
    
    return new_motion_out, M_new_motion_out


if __name__ == "__main__":

    tgt_dir_name = '/ailab/user/henantian/code/ConFiMo/pulse_data'
    os.makedirs(tgt_dir_name, exist_ok=True)

    index_file = pd.read_csv('/ailab/user/henantian/code/ConFiMo/index.csv')
    total_amount = index_file.shape[0]

    for i in tqdm(range(total_amount)):
        src_path = index_file.loc[i]['source_path']
        tgt_file_name = index_file.loc[i]['new_name'].replace('.npy', '.pkl')
        if "humanact12" in src_path:
            src_path = src_path.replace('pose_data', 'amass_data').replace('.npy', '.pkl')
            new_motion_out, M_new_motion_out = store_motion_humanact12(src_path)
        else:
            continue    # Below has been excuted before
            src_path = index_file.loc[i]['source_path'].replace('pose_data', 'amass_data').replace('.npy', '.npz')
            new_motion_out, M_new_motion_out = store_motion(src_path)
        
        joblib.dump(new_motion_out, join(tgt_dir_name, tgt_file_name))
        joblib.dump(M_new_motion_out, join(tgt_dir_name, "M" + tgt_file_name))
    





