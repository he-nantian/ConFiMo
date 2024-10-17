import isaacgym
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

from isaacgym import gymapi
from isaacgym import gymutil
from phc.utils.config import SIM_TIMESTEP

from phc.utils.flags import flags

from phc.utils.motion_lib_smpl import MotionLibSMPL
from easydict import EasyDict
from phc.utils.motion_lib_base import FixHeightMode

from pdb import set_trace as st
from phc.utils import torch_utils
from isaacgym.torch_utils import *
import shutil


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


def parse_sim_params(cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = cfg.sim.slices
    
    if cfg.sim.use_flex:
        if cfg.sim.pipeline in ["gpu"]:
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.use_flex.shape_collision_margin = 0.01
        sim_params.use_flex.num_outer_iterations = 4
        sim_params.use_flex.num_inner_iterations = 10
    else : # use gymapi.SIM_PHYSX
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
        sim_params.physx.num_subscenes = cfg.sim.subscenes
        sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
        
    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if not cfg.sim.use_flex and cfg.sim.physx.num_threads > 0:
        sim_params.physx.num_threads = cfg.sim.physx.num_threads
    
    return sim_params


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


def load_encoder_decoder(cfg, device):

    ######## dimension of PULSE ########
    # s^p: 358    s^g: 576    z: 32    a: 69

    ## encoder
    # encoder:                   934 --> 160
    # z_mu/z_logvar:             160 --> 32

    ## decoder
    # decoder:                   390 --> 69
    # z_prior:                   358 --> 512
    # z_prior_mu/z_prior_logvar: 512 --> 32

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

        pose_aa = pose_aa.reshape(N, 24, 3)[:, left_to_right_index]
        pose_aa[..., 1] *= -1
        pose_aa = pose_aa.reshape(N, 72)

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

    # st()

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

        pose_aa = pose_aa.reshape(N, 24, 3)[:, left_to_right_index]
        pose_aa[..., 1] *= -1
        pose_aa = pose_aa.reshape(N, 72)

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


def remove_base_rot(quat):
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat)) #SMPL
    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))


def compute_obs(body_pos, body_rot, body_vel, body_ang_vel, smpl_params, limb_weight_params, local_root_obs, root_height_obs, upright, has_smpl_params, has_limb_weight_params):
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]
    
    if has_smpl_params:
        obs_list.append(smpl_params)
        
    if has_limb_weight_params:
        obs_list.append(limb_weight_params)

    obs = torch.cat(obs_list, dim=-1)
    return obs


def compute_im_obs(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, time_steps, upright):
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    body_rot[:, None].repeat_interleave(time_steps, 1)
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))


    diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3) - body_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    

    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * 24 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, time_steps, -1))  #  1 * timestep * 24 * 6
    obs.append(diff_local_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(diff_local_ang_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_rot.view(B, time_steps, -1))  # timestep  * 24 * 6

    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs


def motion2obs(current_states, agent):
    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, \
        limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
            current_states["root_pos"], current_states["root_rot"], current_states["dof_pos"], current_states["root_vel"], current_states["root_ang_vel"], current_states["dof_vel"], current_states["motion_bodies"], \
                current_states["motion_limb_weights"], current_states["motion_aa"], current_states["rg_pos"], current_states["rb_rot"], current_states["body_vel"], current_states["body_ang_vel"]
    return compute_obs(rb_pos, rb_rot, body_vel, body_ang_vel, smpl_params, limb_weights, agent._local_root_obs, agent._root_height_obs, agent._has_upright_start, agent._has_shape_obs, agent._has_limb_weight_obs)


if __name__ == "__main__":

    # Custom
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

    # # Data preprocessing
    # tgt_dir_name = '/ailab/user/henantian/code/ConFiMo/pulse_data/motion'
    # os.makedirs(tgt_dir_name, exist_ok=True)

    # index_file = pd.read_csv('/ailab/user/henantian/code/ConFiMo/index.csv')
    # total_amount = index_file.shape[0]

    # for i in tqdm(range(total_amount)):
    #     src_path = index_file.loc[i]['source_path']
    #     tgt_file_name = index_file.loc[i]['new_name'].replace('.npy', '.pkl')
    #     if "humanact12" in src_path:
    #         src_path = src_path.replace('pose_data', 'amass_data').replace('.npy', '.pkl')
    #         new_motion_out, M_new_motion_out = store_motion_humanact12(src_path)
    #     else:
    #         # continue    # Below has been excuted before
    #         src_path = index_file.loc[i]['source_path'].replace('pose_data', 'amass_data').replace('.npy', '.npz')
    #         new_motion_out, M_new_motion_out = store_motion(src_path)
        
    #     joblib.dump(new_motion_out, join(tgt_dir_name, tgt_file_name))
    #     joblib.dump(M_new_motion_out, join(tgt_dir_name, "M" + tgt_file_name))
    
    # motion_set = {}
    # save_dir_name = '/ailab/user/henantian/code/ConFiMo/pulse_data'
    # for filename in tqdm(os.listdir(tgt_dir_name)):
    #     key = filename.split('.')[0]
    #     motion_set[key] = join(tgt_dir_name, filename)
    # joblib.dump(motion_set, join(save_dir_name, 'motion_set.pkl'))
    # logger.info('data saved')

    # # Motion clean
    # logger.info("Removing motion file with less than 10 frames...")
    # tmp_dir = "/ailab/user/henantian/code/ConFiMo/pulse_data/tiny_motion"
    # os.makedirs(tmp_dir, exist_ok=True)
    # motion_path_storer = joblib.load(cfg.MOTION_DIR)
    # key_lst = []
    # for motion_key, motion_path in tqdm(motion_path_storer.items()):
    #     tmp_motion = joblib.load(motion_path)
    #     if tmp_motion['pose_aa'].shape[0] < 10:
    #         key_lst.append(motion_key)
    #         shutil.move(motion_path, tmp_dir)
    #         print(f"Motion {motion_key} has been moved!")
    # print("Updating motion_path_storer...")
    # for k in key_lst:
    #     val = motion_path_storer.pop(k)
    #     print(val)
    # joblib.dump(motion_path_storer, cfg.MOTION_DIR)

    # Motion imitation
    encoder, decoder = load_encoder_decoder(cfg, device)
    logger.info('PULSE loaded')

    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path,  flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d = \
        cfg.debug, cfg.follow, False, False, False, False, False, True, cfg.server_mode, False, False, cfg.im_eval, cfg.no_virtual_display, cfg.render_o3d

    flags.test = True
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    sim_params = parse_sim_params(cfg)
    agent = HumanoidZ(cfg, sim_params, gymapi.SIM_PHYSX, "cuda", 0, cfg.headless)
    agent.initialize_z_models(encoder, decoder)
    obs_mean, obs_std = agent.running_mean.float(), torch.sqrt(agent.running_var.float())
    
    env_ids = torch.arange(agent.num_envs, dtype=torch.long, device=device)
    agent.reset(env_ids)

    motion_lib_cfg = EasyDict({
        "motion_file": cfg.MOTION_DIR,
        "device": device,
        "fix_height": FixHeightMode.full_fix,
        "min_length": -1,
        "max_length": -1,
        "im_eval": flags.im_eval,
        "multi_thread": True ,
        "smpl_type": 'smpl',
        "randomrize_heading": True
    })
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    # st()
    num_motions = motion_lib._num_unique_motions
    num_envs = agent.num_envs
    start_idxes = range(0, num_motions, num_envs)
    logger.info(f"num_motions: {num_motions}\nnum_envs: {num_envs}\nmotion sampling iterations: {len(start_idxes)}")

    latent_dir = cfg.latent_dir
    os.makedirs(latent_dir, exist_ok=True)
    
    for process_id, start_idx in enumerate(start_idxes):

        motion_lib.load_motions(skeleton_trees=agent.skeleton_trees, 
                                gender_betas=agent.humanoid_shapes.cpu(), 
                                limb_weights=agent.humanoid_limb_and_weights.cpu(), 
                                random_sample=False, 
                                start_idx=start_idx,
                                max_len=-1, 
                                num_jobs=1)
        dts = motion_lib._motion_dt
        total_lengths = motion_lib._motion_lengths
        num_frames = motion_lib._motion_num_frames
        max_num_frame = num_frames.max()

        current_keys = motion_lib.curr_motion_keys
        if_continue = True
        for key in current_keys:
            if not os.path.exists(join(latent_dir, f"{key}.npy")):
                if_continue = False
                break
        if if_continue:
            print(f"Already processed:\n{current_keys}")
            continue

        timesteps = torch.zeros(size=(agent.num_envs,), dtype=torch.float32, device=device)
        current_states = motion_lib.get_motion_state(motion_lib.motion_ids, timesteps)

        ## Set the state of robots consistency with the initial motion
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, \
            limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                current_states["root_pos"], current_states["root_rot"], current_states["dof_pos"], current_states["root_vel"], current_states["root_ang_vel"], current_states["dof_vel"], current_states["motion_bodies"], \
                    current_states["motion_limb_weights"], current_states["motion_aa"], current_states["rg_pos"], current_states["rb_rot"], current_states["body_vel"], current_states["body_ang_vel"]
        agent._set_env_state(env_ids=env_ids, root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos, root_vel=root_vel, root_ang_vel=root_ang_vel, dof_vel=dof_vel, rigid_body_pos=rb_pos, rigid_body_rot=rb_rot, rigid_body_vel=body_vel, rigid_body_ang_vel=body_ang_vel)
        agent._reset_env_tensors(env_ids)
        agent.obs_buf = agent._compute_observations(env_ids)
        
        # masks = torch.ones(size=(agent.num_envs,), dtype=torch.float32, device=device)

        # ## For test
        # from matplotlib import pyplot as plt
        # diff, diff2 = [], []

        z_seqs = np.zeros((num_envs, max_num_frame - 2, 32))

        for frame_id, next_frame in enumerate(range(1, max_num_frame - 1)): # As we cannot compute the velocity of the last frame 
            
            print(f"########## Motion sampling procedure: {process_id}/{len(start_idxes)} ##########")
            print(f"##### Imitation procedure: {frame_id}/{max_num_frame-2} #####")

            masks = num_frames > next_frame + 1    # (num_envs,)
            timesteps = masks * (timesteps + dts)
            next_states = motion_lib.get_motion_state(motion_lib.motion_ids, timesteps)

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, \
                limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                    next_states["root_pos"], next_states["root_rot"], next_states["dof_pos"], next_states["root_vel"], next_states["root_ang_vel"], next_states["dof_vel"], next_states["motion_bodies"], \
                        next_states["motion_limb_weights"], next_states["motion_aa"], next_states["rg_pos"], next_states["rb_rot"], next_states["body_vel"], next_states["body_ang_vel"]

            # # For test
            # next_obs_im = motion2obs(next_states, agent)

            im_obs = compute_im_obs(
                root_pos=agent._rigid_body_pos[:, 0], 
                root_rot=agent._rigid_body_rot[:, 0], 
                body_pos=agent._rigid_body_pos, 
                body_rot=agent._rigid_body_rot, 
                body_vel=agent._rigid_body_vel, 
                body_ang_vel=agent._rigid_body_ang_vel, 
                ref_body_pos=rb_pos.unsqueeze(1), 
                ref_body_rot=rb_rot.unsqueeze(1), 
                ref_body_vel=body_vel.unsqueeze(1), 
                ref_body_ang_vel=body_ang_vel.unsqueeze(1), 
                time_steps=1, 
                upright=upright_start
            )

            encoder_input = (torch.cat([agent.obs_buf, im_obs], dim=-1) - obs_mean) / (obs_std + 1e-05)
            encoder_latent = encoder.encoder(encoder_input)

            action_z = encoder.z_mu(encoder_latent)    # (num_envs, 32)

            ## For test
            # action_mu, action_logvar = encoder.z_mu(encoder_latent), encoder.z_logvar(encoder_latent)
            # action_z = action_mu + torch.randn_like(action_mu) * torch.exp(action_logvar * 0.5)
            
            # action_z = decoder.z_prior_mu(decoder.z_prior((agent.obs_buf - obs_mean[:358]) / (obs_std[:358] + 1e-5)))

            # action_z = -encoder.z_mu(encoder_latent)

            # action_z = torch.randn(size=(1, 32), dtype=torch.float32, device=device)

            # action_z = torch.zeros(size=(1, 32), dtype=torch.float32, device=device)

            agent.step_z(action_z)

            true_idx = [i for i, val in enumerate(masks) if val]
            z_seqs[true_idx, frame_id] = action_z[true_idx].detach().numpy()
        
        for key, z_seq, num_frame in zip(current_keys, z_seqs, num_frames):
            np.save(join(latent_dir, f"{key}.npy"), z_seq[:num_frame-2])
        
    logger.info("Latent dataset has been constructed!")
        
        # st()

        #     # For test
        #     next_obs = agent._compute_observations(env_ids)
        #     temp = float((next_obs_im[0]-next_obs[0]).abs().mean().detach())
        #     diff.append(temp)
        #     temp2 = float((next_obs[0][:1]).abs().mean().detach())
        #     diff2.append(temp2)
        
        # # st()
        # plt.figure()
        # plt.plot(diff)
        # plt.savefig('tmp_neg.png')
        # plt.close()
        # plt.figure()
        # plt.plot(diff2)
        # plt.savefig('tmp_neg2.png')
        # plt.close()
        # break
            



    # st()

    # init_state = agent.reset()
    # print(init_state.size(), init_state)
    # step_state = agent.step_z(torch.zeros(size=(2, 32)).cuda())
    # print(step_state.size(), step_state)
    # print(agent.get_obs_size())
    # print(agent.get_action_size())
    # print(agent.get_dof_action_size())
    # print(agent.get_num_actors_per_env())