import os
import numpy as np
import joblib
from os.path import join as pjoin
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.getcwd())


index_path = './index.csv'
index_file = pd.read_csv(index_path)
total_amount = index_file.shape[0]

count = true_count = 0
data_raw = []

for i in tqdm(range(total_amount)):
    source_path = index_file.loc[i]['source_path'].replace('pose_data', 'amass_data')
    if 'humanact12/humanact12' in source_path:
        count += 1
        if os.path.exists(source_path):
            true_count += 1
            data_raw.append((np.load(source_path), source_path.split('/')[-1].split('.')[0]))
        else:
            print(f"{source_path} not exists!")

print(count, true_count)

data = joblib.load('/ailab/user/henantian/code/ConFiMo/amass_data/humanact12poses.pkl')
data_clean = []
for poses, joints3D in zip(data['poses'], data['joints3D']):
    data_clean.append((joints3D, poses))

joblib.dump(data_raw, './data_raw.pkl', compress=3)
joblib.dump(data_clean, './data_clean.pkl', compress=3)

match_results = {
    'idx_raw': [], 
    'idx_clean': [], 
    'joints3D_raw': [], 
    'joints3D_clean': [], 
    'poses': [], 
    'keyword': [], 
    'min_dist': []
}

for idx_clean, (joints3D_clean, poses) in tqdm(enumerate(data_clean)):
    min_dist = np.inf
    idx_raw, joints3D_raw, keyword = None, None, None
    for idx, (joints3D, kw) in enumerate(data_raw):
        if joints3D.shape != joints3D_clean.shape:
            continue
        cur_dist = np.mean(np.abs(joints3D_clean - joints3D))
        if cur_dist < min_dist:
            min_dist = cur_dist
            idx_raw = idx
            keyword = kw
            joints3D_raw = joints3D
    if idx_raw is None:
        print(f"Any data not match the shape of {str(joints3D_clean.shape)} in idx {idx_clean}!")
    match_results['idx_raw'].append(idx_raw)
    match_results['idx_clean'].append(idx_clean)
    match_results['joints3D_raw'].append(joints3D_raw)
    match_results['joints3D_clean'].append(joints3D_clean)
    match_results['poses'].append(poses)
    match_results['keyword'].append(keyword)
    match_results['min_dist'].append(min_dist)

joblib.dump(match_results, './match_results.pkl', compress=3)



