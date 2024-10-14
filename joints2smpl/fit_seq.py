from __future__ import print_function, division
import argparse
import torch
import os,sys
from os import walk, listdir
from os.path import isfile, join
import numpy as np
import joblib
import smplx
import trimesh
import h5py
from tqdm import tqdm

from src.smplify import SMPLify3D
from src import config

# parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1,
                    help='input batch size')
parser.add_argument('--num_smplify_iters', type=int, default=2000,
                    help='num of smplify iters')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--gpu_ids', type=int, default=0,
                    help='choose gpu ids')
parser.add_argument('--num_joints', type=int, default=24,
                    help='joint number')
parser.add_argument('--joint_category', type=str, default="AMASS",
                    help='use correspondence')
parser.add_argument('--fix_foot', type=str, default="False",
                    help='fix foot or not')
parser.add_argument('--data_file', type=str, default="/ailab/user/henantian/code/ConFiMo/data_raw.pkl",
                    help='data')
parser.add_argument('--save_folder', type=str, default="/ailab/user/henantian/code/ConFiMo/amass_data/humanact12/humanact12/",
                    help='results save folder')

parser.add_argument('--num_data', type=int, default=100, help='number of data to process')

parser.add_argument('--start_idx', type=int, default=0, help='start index')  # total_num: 1191

opt = parser.parse_args()
print(opt)

# ---load predefined something
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
print(config.SMPL_MODEL_DIR)
smplmodel = smplx.create(config.SMPL_MODEL_DIR, 
                         model_type="smpl", gender="neutral", ext="pkl",
                         batch_size=opt.batchSize).to(device)

# ## --- load the mean pose as original ---- 
smpl_mean_file = config.SMPL_MEAN_FILE

file = h5py.File(smpl_mean_file, 'r')
init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()
cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device)
#
pred_pose = torch.zeros(opt.batchSize, 72).to(device)
pred_betas = torch.zeros(opt.batchSize, 10).to(device)
pred_cam_t = torch.zeros(opt.batchSize, 3).to(device)
keypoints_3d = torch.zeros(opt.batchSize, opt.num_joints, 3).to(device)

dataset = joblib.load(opt.data_file)

z_min = min([np.min(data[0][..., 1]) for data in dataset])
print(f"z_min: {z_min}")

total_num = len(dataset)
end_idx = min(opt.start_idx + opt.num_data, total_num)
dataset = dataset[opt.start_idx:end_idx]

for data, purename in dataset:

	# dir_save = os.path.join(opt.save_folder, purename)
	# if not os.path.isdir(dir_save):
	# 	os.makedirs(dir_save, exist_ok=True)

	# run the whole seqs
	num_seqs = data.shape[0]

	trans_matrix = np.array(
		[[0, -1, 0], 
		[0, 0, 1], 
		[1, 0, 0]]
	)

	for idx in tqdm(range(num_seqs)):
		#print(idx)

		joints3d = np.dot(data[idx], trans_matrix) #*1.2 #scale problem [check first]	
		joints3d[:, 2] = joints3d[:, 2] - z_min
		keypoints_3d[0, :, :] = torch.Tensor(joints3d).to(device).float()

		if idx == 0:
			pred_betas[0, :] = init_mean_shape
			pred_pose[0, :] = init_mean_pose
			pred_cam_t[0, :] = cam_trans_zero
		else:
			pred_betas[0, :] = torch.from_numpy(param['beta']).unsqueeze(0).float()
			pred_pose[0, :] = torch.from_numpy(param['pose']).unsqueeze(0).float()
			pred_cam_t[0, :] = torch.from_numpy(param['cam']).unsqueeze(0).float()
			
		if opt.joint_category =="AMASS":
			confidence_input =  torch.ones(opt.num_joints)
			# make sure the foot and ankle
			if opt.fix_foot == True:
				confidence_input[7] = 1.5
				confidence_input[8] = 1.5
				confidence_input[10] = 1.5
				confidence_input[11] = 1.5
		else:
			print("Such category not settle down!")
		
		# # #-------------initialize SMPLify
		num_smplify_iters = opt.num_smplify_iters if idx == 0 else int(opt.num_smplify_iters / 5)
		smplify = SMPLify3D(smplxmodel=smplmodel,
							batch_size=opt.batchSize,
							joints_category=opt.joint_category,
							num_iters=num_smplify_iters,
							device=device, 
							use_lbfgs=False)
		#print("initialize SMPLify3D done!")
		# ----- from initial to fitting -------
		new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
		new_opt_cam_t, new_opt_joint_loss = smplify(
													pred_pose.detach(),
													pred_betas.detach(),
													pred_cam_t.detach(),
													keypoints_3d,
													conf_3d=confidence_input.to(device),
													seq_ind=idx
													)

		# # -- save the results to ply---
		new_opt_cam_t = new_opt_cam_t[0]
		# outputp = smplmodel(betas=new_opt_betas, global_orient=new_opt_pose[:, :3], body_pose=new_opt_pose[:, 3:],
		# 					transl=new_opt_cam_t, return_verts=True)
		# mesh_p = trimesh.Trimesh(vertices=outputp.vertices.detach().cpu().numpy().squeeze(), faces=smplmodel.faces, process=False)
		# mesh_p.export(dir_save + "/" + "%04d"%idx + ".ply")
		
		# save the pkl
		param = {}
		param['beta'] = new_opt_betas.detach().cpu().numpy()
		param['pose'] = new_opt_pose.detach().cpu().numpy()
		param['cam'] = new_opt_cam_t.detach().cpu().numpy()
		# joblib.dump(param, dir_save + "/" + "%04d"%idx + ".pkl", compress=3)

		if idx == 0:
			params = {}
			params['beta'] = new_opt_betas.detach().cpu().numpy()
			params['pose'] = new_opt_pose.detach().cpu().numpy()
			params['cam'] = new_opt_cam_t.detach().cpu().numpy()
		else:
			params['beta'] = np.concatenate((params['beta'], new_opt_betas.detach().cpu().numpy()))
			params['pose'] = np.concatenate((params['pose'], new_opt_pose.detach().cpu().numpy()))
			params['cam'] = np.concatenate((params['cam'], new_opt_cam_t.detach().cpu().numpy()))

	joblib.dump(params, opt.save_folder + f"{purename}.pkl", compress=3)
