import os
from argparse import ArgumentParser
from glob import glob
from warnings import showwarning
from argoverse.visualization.mayavi_utils import Figure

import cv2
import numpy as np
import torch
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from fiery.data import prepare_dataloaders
from fiery.config import get_parser, get_cfg
from fiery.utils.geometry import mat2pose_vec
from fiery.trainer import TrainingModule
from fiery.utils.network import NormalizeInverse
from fiery.utils.instance import predict_instance_segmentation_and_trajectories
from fiery.utils.visualisation import plot_instance_map, generate_instance_colours, make_contour, convert_figure_numpy

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.calibration import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_calibration_config
import argoverse.visualization.visualization_utils as vis_utils

EXAMPLE_DATA_PATH = 'example_data'


def plot_prediction(image, output, cfg):
	# Process predictions
	consistent_instance_seg, matched_centers = predict_instance_segmentation_and_trajectories(output, compute_matched_centers=True)

	# Plot future trajectories
	unique_ids = torch.unique(consistent_instance_seg[0, 0]).cpu().long().numpy()[1:]
	instance_map = dict(zip(unique_ids, unique_ids))
	instance_colours = generate_instance_colours(instance_map)
	vis_image = plot_instance_map(consistent_instance_seg[0, 0].cpu().numpy(), instance_map)
	trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
	for instance_id in unique_ids:
		path = matched_centers[instance_id]
		for t in range(len(path) - 1):
			color = instance_colours[instance_id].tolist()
			cv2.line(trajectory_img, tuple(path[t]), tuple(path[t + 1]),
					 color, 4)

	# Overlay arrows
	temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 1.0)
	mask = ~ np.all(trajectory_img == 0, axis=2)
	vis_image[mask] = temp_img[mask]

	# Plot present RGB frames and predictions
	val_w = 2.99
	cameras = cfg.IMAGE.NAMES
	image_ratio = cfg.IMAGE.FINAL_DIM[0] / cfg.IMAGE.FINAL_DIM[1]
	val_h = val_w * image_ratio
	fig = plt.figure(figsize=(4 * val_w, 2 * val_h))
	width_ratios = (val_w, val_w, val_w, val_w)
	gs = mpl.gridspec.GridSpec(3, 4, width_ratios=width_ratios)
	gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

	denormalise_img = torchvision.transforms.Compose(
		(NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		 torchvision.transforms.ToPILImage(),)
	)
	for imgi, img in enumerate(image[0, -1]):
		ax = plt.subplot(gs[imgi // 3, imgi % 3])
		showimg = denormalise_img(img.cpu())
		if imgi > 2:
			showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)

		# plt.annotate(cameras[imgi].replace('_', ' ').replace('CAM ', ''), (0.01, 0.87), c='white',
		#              xycoords='axes fraction', fontsize=14)
		plt.imshow(showimg)
		plt.axis('off')

	ax = plt.subplot(gs[:, 3])
	plt.imshow(make_contour(vis_image[::-1, ::-1]))
	plt.axis('off')

	plt.draw()
	figure_numpy = convert_figure_numpy(fig)
	plt.close()
	return figure_numpy

def get_input_data_timestamp(loader, frame_idx, log_idx)-> None:
	normilize_image = torchvision.transforms.Compose(
		(torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),)
	)
	denormalise_img = torchvision.transforms.Compose(
		(NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		torchvision.transforms.ToPILImage(),)
	)

	images = []
	intrinsics_matrices = [] 
	extrinsics_matrices = []
	for camera in loader.CAMERA_LIST[:-2]:
			image = loader.get_image_sync(frame_idx, camera=camera, load=True)
			image = cv2.resize(image, (480,224))
			image = normilize_image(image).unsqueeze(0)
			images.append(image)

			calibration = loader.get_calibration(camera, log_idx)

			intrinsics_matrix = calibration.K[0:3, 0:3]
			intrinsics = torch.from_numpy(intrinsics_matrix).unsqueeze(0)
			intrinsics_matrices.append(intrinsics)

			# from camera_frame to egovehicle_frame
			extrinsics_matrix = np.linalg.inv(calibration.extrinsic)
			# extrinsics_matrix = calibration.extrinsic
			extrinsics = torch.from_numpy(extrinsics_matrix).unsqueeze(0)
			extrinsics_matrices.append(extrinsics)

	return images, intrinsics_matrices, extrinsics_matrices


def visualise(args):
	checkpoint_path = args.checkpoint
	# cfg = get_cfg(args.config)

	# trainer = TrainingModule(cfg)
	trainer = TrainingModule.load_from_checkpoint(checkpoint_path)
	# trainer.cfg.TIME_RECEPTIVE_FIELD = 1
	# trainer.cfg.N_FUTURE_FRAMES = 0
	# trainer.cfg.MODEL.TEMPORAL_MODEL.NAME = 'identity'
	# trainer.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE = False

	# trainer.cfg.

	device = torch.device('cuda:0')
	# device = torch.device('cpu')

	trainer = trainer.to(device)
	trainer.eval()

	root_dir = os.path.join(os.getcwd(), 'argoverse-api/data/argoverse-tracking/sample')
	tracking_loader = ArgoverseTrackingLoader(root_dir)

	seconds_in_past = 1
	frame_rate = 10
	frames = [i for i in range(0, seconds_in_past * frame_rate + 1, 5)]
	# frames = [100]
	images_past = []
	extrinsics = []
	intrinsics = []
	relative_egomotion = []
	for i, frame_idx in enumerate(frames):
		images_timestamp, intrinsics_timestamp, extrinsics_timestamp = \
			get_input_data_timestamp(tracking_loader, frame_idx, tracking_loader.log_list[0])

		# recast images_timestamp, intrs, extrs to tensors 
		# of size(n_cameras, c, h, w) and append to lists
		images_timestamp_tensor = torch.cat(images_timestamp, dim=0).unsqueeze(0).unsqueeze(0)
		images_past.append(images_timestamp_tensor)

		intrinsics_timestamp_tensor = torch.cat(intrinsics_timestamp, dim=0).unsqueeze(0).unsqueeze(0)
		intrinsics.append(intrinsics_timestamp_tensor)
		
		extrinsics_timestamp_tensor = torch.cat(extrinsics_timestamp, dim=0).unsqueeze(0).unsqueeze(0)
		extrinsics.append(extrinsics_timestamp_tensor)

		#egomotion between 2 adjacent poses (frames)
		egomotion = np.eye(4, dtype=np.float32)
		if i < len(frames) - 1:
			city_to_ego_present_T = tracking_loader.get_pose(frames[i]).transform_matrix
			city_to_ego_next_T = tracking_loader.get_pose(frames[i+1]).transform_matrix
			egomotion = np.linalg.inv(city_to_ego_present_T).dot(city_to_ego_next_T)
		egomotion = torch.Tensor(egomotion).float()
		pose_6dof_vec = mat2pose_vec(egomotion).unsqueeze(0).unsqueeze(0)
		relative_egomotion.append(pose_6dof_vec)

	images_past = torch.cat(images_past, dim=1)
	intrinsics = torch.cat(intrinsics, dim=1)
	extrinsics = torch.cat(extrinsics, dim=1)
	relative_egomotion = torch.cat(relative_egomotion, dim=1)
	print(f'images_past.shape: {images_past.shape}\nintrinsic_past.shape: {intrinsics.shape}\nextrinsics_past.shape: {extrinsics.shape}\npose_past.shape: {relative_egomotion.shape}')


	images_past = images_past.float().to(device)
	intrinsics = intrinsics.float().to(device)
	extrinsics = extrinsics.float().to(device)
	relative_egomotion = relative_egomotion.float().to(device)

	with torch.no_grad():
		output = trainer.model(images_past, intrinsics, extrinsics, relative_egomotion)

	figure_numpy = plot_prediction(images_past, output, trainer.cfg)
	showimg = Image.fromarray(figure_numpy)
	plt.imshow(showimg)
	plt.waitforbuttonpress()
	plt.axis('off')


def test_extrinsics():
	root_dir = os.path.join(os.getcwd(), 'argoverse-api/data/argoverse-tracking/sample')
	argoverse_loader = ArgoverseTrackingLoader(root_dir)
	argoverse_data = argoverse_loader.get(argoverse_loader.log_list[0])
	idx = 100
	f,ax = vis_utils.make_grid_ring_camera(argoverse_data, idx)
	plt.show()
	return


if __name__ == '__main__':
	parser = ArgumentParser(description='Fiery visualisation')
	parser.add_argument('--checkpoint', default='./fiery/checkpoints/fiery.ckpt', type=str, help='path to checkpoint')
	parser.add_argument('--config', default='fiery/configs/argoverse/single_timeframe.yml', type=str, help='path to config file')
	args = parser.parse_args()

	visualise(args)
	# test_extrinsics()




