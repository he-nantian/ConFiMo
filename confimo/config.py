import os
import importlib
from typing import Type, TypeVar
from argparse import ArgumentParser

from omegaconf import OmegaConf, DictConfig
from typing import List

def get_module_config(cfg_model: DictConfig, paths: List) -> DictConfig:
    files = [os.path.join('./configs/modules', p+'.yaml') for p in paths]
    for file in files:
        assert os.path.exists(file), f'{file} is not exists.'
        with open(file, 'r') as f:
            cfg_model.merge_with(OmegaConf.load(f))
    return cfg_model


def get_obj_from_str(string: str, reload: bool = False) -> Type:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: DictConfig) -> TypeVar:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def parse_args() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--cfg", "-c", type=str, required=True, help="The main config file")
    parser.add_argument('--example', type=str, required=False, help="The input texts and lengths with txt format")
    parser.add_argument('--no-plot', action="store_true", required=False, help="Whether to plot the skeleton-based motion")
    parser.add_argument('--replication', type=int, default=1, help="The number of replications of sampling")
    parser.add_argument('--vis', type=str, default="tb", choices=['tb', 'swanlab'], help="The visualization backends: tensorboard or swanlab")
    
    parser.add_argument("--vis_gpu", "-vg", type=str, default="", help="Visible GPU")
    parser.add_argument("--folder", "-f", type=str, default="", help="Folder of the experiment")
    parser.add_argument("--name", "-n", type=str, default="", help="Name of the experiment")

    # argument for MotionLCM
    parser.add_argument("--num_sampling", "-ns", type=int, default=1, help="n-step inference of MotionLCM")

    # argument for training
    parser.add_argument("--pretrain", "-pt", type=str, default="", help="Pretrained path for training")

    # arguments for test
    parser.add_argument("--test_checkpoint", "-tcp", type=str, default="", help="Checkpoint for evaluating its metrics")
    
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    cfg_model = get_module_config(cfg.model, cfg.model.target)
    if args.cfg == "configs/motionlcm_t2m.yaml":
        cfg_model.scheduler.num_inference_timesteps = args.num_sampling
        # import pdb; pdb.set_trace()
    
    cfg = OmegaConf.merge(cfg, cfg_model)

    if args.cfg == "configs/pulse.yaml":
        cfg_env = get_module_config(cfg.env, cfg.env.target)
        cfg_robot = get_module_config(cfg.robot, cfg.robot.target)
        cfg_sim = get_module_config(cfg.sim, cfg.sim.target)
        cfg = OmegaConf.merge(cfg, cfg_env, cfg_robot, cfg_sim)
    # import pdb; pdb.set_trace()

    cfg.example = args.example
    cfg.no_plot = args.no_plot
    cfg.replication = args.replication
    cfg.vis = args.vis
    
    cfg.vis_gpu = args.vis_gpu if args.vis_gpu else None
    cfg.NAME = args.name if args.name else cfg.NAME
    cfg.FOLDER = args.folder if args.folder else cfg.FOLDER

    if args.cfg in ["configs/mld_t2m.yaml", "configs/motionlcm_t2m.yaml"]:
        cfg.TRAIN.PRETRAINED = args.pretrain if args.pretrain else cfg.TRAIN.PRETRAINED
        cfg.TEST.CHECKPOINTS = args.test_checkpoint if args.test_checkpoint else cfg.TEST.CHECKPOINTS
    else:
        print("Not implemented yet!")
    return cfg
