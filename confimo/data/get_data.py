from os.path import join as pjoin
from typing import Callable, Optional

import numpy as np

from omegaconf import DictConfig

from .humanml.utils.word_vectorizer import WordVectorizer
from .HumanML3D import HumanML3DDataModule
from .Kit import KitDataModule
from .base import BaseDataModule
from .utils import mld_collate


def get_mean_std(phase: str, cfg: DictConfig, dataset_name: str) -> tuple:
    name = "t2m" if dataset_name == "humanml3d" else dataset_name
    assert name in ["t2m", "kit"]
    if phase in ["val"]:
        if name == 't2m':
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD01", "meta")
        elif name == 'kit':
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD005", "meta")
        else:
            raise ValueError("Only support t2m and kit")
        mean = np.load(pjoin(data_root, "mean.npy"))
        std = np.load(pjoin(data_root, "std.npy"))
    else:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))

    return mean, std


def get_WordVectorizer(cfg: DictConfig, phase: str, dataset_name: str) -> Optional[WordVectorizer]:
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None


def get_collate_fn(name: str) -> Callable:
    if name.lower() in ["humanml3d", "kit"]:
        return mld_collate
    else:
        raise NotImplementedError


dataset_module_map = {"humanml3d": HumanML3DDataModule, "kit": KitDataModule}
motion_subdir = {"humanml3d": "new_joint_vecs", "kit": "new_joint_vecs"}


def get_dataset(cfg: DictConfig, phase: str = "train") -> BaseDataModule:
    dataset_name = eval(f"cfg.{phase.upper()}.DATASET")
    if dataset_name.lower() in ["humanml3d", "kit"]:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean, std = get_mean_std(phase, cfg, dataset_name)
        mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
        wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
        collate_fn = get_collate_fn(dataset_name)
        dataset = dataset_module_map[dataset_name.lower()](
            cfg=cfg,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            collate_fn=collate_fn,
            persistent_workers=cfg.TRAIN.PERSISTENT_WORKERS,
            mean=mean,
            std=std,
            mean_eval=mean_eval,
            std_eval=std_eval,
            w_vectorizer=wordVectorizer,
            text_dir=pjoin(data_root, "texts"),
            motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
            max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
            min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
            max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
            unit_length=eval(f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            fps=eval(f"cfg.DATASET.{dataset_name.upper()}.FRAME_RATE"),
            model_kwargs=cfg.model)

    elif dataset_name.lower() in ["humanact12", 'uestc', "amass"]:
        raise NotImplementedError

    cfg.DATASET.NFEATS = dataset.nfeats
    cfg.DATASET.NJOINTS = dataset.njoints
    return dataset
