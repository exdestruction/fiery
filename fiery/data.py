import os
from typing import Dict, List, Tuple, Union

from PIL import Image
from matplotlib.pyplot import get
import numpy as np
import cv2
import torch
import torchvision

import matplotlib as mpl
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset

if __name__ == "__main__":
	import sys
	sys.path.append('../fiery')

from fiery.utils.geometry import (
	resize_and_crop_image,
	update_intrinsics,
	calculate_birds_eye_view_parameters,
	convert_egopose_to_matrix_numpy,
	pose_vec2mat,
	mat2pose_vec,
	invert_matrix_egopose_numpy,
)
from fiery.utils.instance import convert_instance_mask_to_center_and_offset_label
from fiery.utils.lyft_splits import TRAIN_LYFT_INDICES, VAL_LYFT_INDICES
from fiery.config import get_cfg
from fiery.utils.network import NormalizeInverse
from fiery.utils.visualisation import plot_instance_map, generate_instance_colours, make_contour, convert_figure_numpy

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader


class FuturePredictionDataset(torch.utils.data.Dataset):
	def __init__(self, dataset, is_train, cfg):
		self.dataset = dataset
		self.is_train = is_train
		self.cfg = cfg

		self.is_lyft = isinstance(dataset, LyftDataset)

		if self.is_lyft:
			self.dataroot = self.dataset.data_path

		else:
			self.dataroot = self.dataset.dataroot

		self.mode = 'train' if self.is_train else 'val'

		self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES

		self.scenes = self.get_scenes()
		self.ixes = self.prepro()
		self.indices = self.get_indices()

		# Image resizing and cropping
		self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

		# Normalising input images
		self.normalise_image = torchvision.transforms.Compose(
			[torchvision.transforms.ToTensor(),
			 torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)

		# Bird's-eye view parameters
		bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
			cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
		)
		self.bev_resolution, self.bev_start_position, self.bev_dimension = (
			bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
		)

		# Spatial extent in bird's-eye view, in meters
		self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

	def get_scenes(self):

		if self.is_lyft:
			scenes = [row['name'] for row in self.dataset.scene]

			# Split in train/val
			indices = TRAIN_LYFT_INDICES if self.is_train else VAL_LYFT_INDICES
			scenes = [scenes[i] for i in indices]
		else:
			# filter by scene split
			split = {'v1.0-trainval': {True: 'train', False: 'val'},
					 'v1.0-mini': {True: 'mini_train', False: 'mini_val'},}[
				self.dataset.version
			][self.is_train]

			scenes = create_splits_scenes()[split]

		return scenes

	def prepro(self):
		samples = [samp for samp in self.dataset.sample]

		# remove samples that aren't in this split
		samples = [samp for samp in samples if self.dataset.get('scene', samp['scene_token'])['name'] in self.scenes]

		# sort by scene, timestamp (only to make chronological viz easier)
		samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

		return samples

	def get_indices(self):
		indices = []
		for index in range(len(self.ixes)):
			is_valid_data = True
			previous_rec = None
			current_indices = []
			for t in range(self.sequence_length):
				index_t = index + t
				# Going over the dataset size limit.
				if index_t >= len(self.ixes):
					is_valid_data = False
					break
				rec = self.ixes[index_t]
				# Check if scene is the same
				if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
					is_valid_data = False
					break

				current_indices.append(index_t)
				previous_rec = rec

			if is_valid_data:
				indices.append(current_indices)

		return np.asarray(indices)

	def get_resizing_and_cropping_parameters(self):
		original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
		final_height, final_width = self.cfg.IMAGE.FINAL_DIM

		resize_scale = self.cfg.IMAGE.RESIZE_SCALE
		resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
		resized_width, resized_height = resize_dims

		crop_h = self.cfg.IMAGE.TOP_CROP
		crop_w = int(max(0, (resized_width - final_width) / 2))
		# Left, top, right, bottom crops.
		crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

		if resized_width != final_width:
			print('Zero padding left and right parts of the image.')
		if crop_h + final_height != resized_height:
			print('Zero padding bottom part of the image.')

		return {'scale_width': resize_scale,
				'scale_height': resize_scale,
				'resize_dims': resize_dims,
				'crop': crop,
				}

	def get_input_data(self, rec):
		"""
		Parameters
		----------
			rec: nuscenes identifier for a given timestamp

		Returns
		-------
			images: torch.Tensor<float> (N, 3, H, W)
			intrinsics: torch.Tensor<float> (3, 3)
			extrinsics: torch.Tensor(N, 4, 4)
		"""
		images = []
		intrinsics = []
		extrinsics = []
		cameras = self.cfg.IMAGE.NAMES

		# The extrinsics we want are from the camera sensor to "flat egopose" as defined
		# https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
		# which corresponds to the position of the lidar.
		# This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

		# From lidar egopose to world.
		lidar_sample = self.dataset.get('sample_data', rec['data']['LIDAR_TOP'])
		lidar_pose = self.dataset.get('ego_pose', lidar_sample['ego_pose_token'])
		yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
		lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
		lidar_translation = np.array(lidar_pose['translation'])[:, None]
		lidar_to_world = np.vstack([
			np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
			np.array([0, 0, 0, 1])
		])

		for cam in cameras:
			camera_sample = self.dataset.get('sample_data', rec['data'][cam])

			# Transformation from world to egopose
			car_egopose = self.dataset.get('ego_pose', camera_sample['ego_pose_token'])
			egopose_rotation = Quaternion(car_egopose['rotation']).inverse
			egopose_translation = -np.array(car_egopose['translation'])[:, None]
			world_to_car_egopose = np.vstack([
				np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
				np.array([0, 0, 0, 1])
			])

			# From egopose to sensor
			sensor_sample = self.dataset.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
			intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
			sensor_rotation = Quaternion(sensor_sample['rotation'])
			sensor_translation = np.array(sensor_sample['translation'])[:, None]
			car_egopose_to_sensor = np.vstack([
				np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
				np.array([0, 0, 0, 1])
			])
			car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

			# Combine all the transformation.
			# From sensor to lidar.
			lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
			sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

			# Load image
			image_filename = os.path.join(self.dataroot, camera_sample['filename'])
			img = Image.open(image_filename)
			# Resize and crop
			img = resize_and_crop_image(
				img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
			)
			# Normalise image
			normalised_img = self.normalise_image(img)

			# Combine resize/cropping in the intrinsics
			top_crop = self.augmentation_parameters['crop'][1]
			left_crop = self.augmentation_parameters['crop'][0]
			intrinsic = update_intrinsics(
				intrinsic, top_crop, left_crop,
				scale_width=self.augmentation_parameters['scale_width'],
				scale_height=self.augmentation_parameters['scale_height']
			)

			images.append(normalised_img.unsqueeze(0).unsqueeze(0))
			intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
			extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

		images, intrinsics, extrinsics = (torch.cat(images, dim=1),
										  torch.cat(intrinsics, dim=1),
										  torch.cat(extrinsics, dim=1)
										  )

		return images, intrinsics, extrinsics

	def _get_top_lidar_pose(self, rec):
		egopose = self.dataset.get('ego_pose', self.dataset.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
		trans = -np.array(egopose['translation'])
		yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
		rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
		return trans, rot

	def get_birds_eye_view_label(self, rec, instance_map):
		translation, rotation = self._get_top_lidar_pose(rec)
		segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
		# Background is ID 0
		instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
		z_position = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
		attribute_label = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

		for annotation_token in rec['anns']:
			# Filter out all non vehicle instances
			annotation = self.dataset.get('sample_annotation', annotation_token)

			if not self.is_lyft:
				# NuScenes filter
				if 'vehicle' not in annotation['category_name']:
					continue
				if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1:
					continue
			else:
				# Lyft filter
				if annotation['category_name'] not in ['bus', 'car', 'construction_vehicle', 'trailer', 'truck']:
					continue

			if annotation['instance_token'] not in instance_map:
				instance_map[annotation['instance_token']] = len(instance_map) + 1
			instance_id = instance_map[annotation['instance_token']]

			if not self.is_lyft:
				instance_attribute = int(annotation['visibility_token'])
			else:
				instance_attribute = 0

			poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
			cv2.fillPoly(instance, [poly_region], instance_id)
			cv2.fillPoly(segmentation, [poly_region], 1.0)
			cv2.fillPoly(z_position, [poly_region], z)
			cv2.fillPoly(attribute_label, [poly_region], instance_attribute)

		return segmentation, instance, z_position, instance_map, attribute_label

	def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation):
		box = Box(
			instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
		)
		box.translate(ego_translation)
		box.rotate(ego_rotation)

		pts = box.bottom_corners()[:2].T
		pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
		pts[:, [1, 0]] = pts[:, [0, 1]]

		z = box.bottom_corners()[2, 0]
		return pts, z

	def get_label(self, rec, instance_map):
		segmentation_np, instance_np, z_position_np, instance_map, attribute_label_np = \
			self.get_birds_eye_view_label(rec, instance_map)
		segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
		instance = torch.from_numpy(instance_np).long().unsqueeze(0)
		z_position = torch.from_numpy(z_position_np).float().unsqueeze(0).unsqueeze(0)
		attribute_label = torch.from_numpy(attribute_label_np).long().unsqueeze(0).unsqueeze(0)

		return segmentation, instance, z_position, instance_map, attribute_label

	def get_future_egomotion(self, rec, index):
		rec_t0 = rec

		# Identity
		future_egomotion = np.eye(4, dtype=np.float32)

		if index < len(self.ixes) - 1:
			rec_t1 = self.ixes[index + 1]

			if rec_t0['scene_token'] == rec_t1['scene_token']:
				egopose_t0 = self.dataset.get(
					'ego_pose', self.dataset.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
				)
				egopose_t1 = self.dataset.get(
					'ego_pose', self.dataset.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
				)

				egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
				egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

				future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
				future_egomotion[3, :3] = 0.0
				future_egomotion[3, 3] = 1.0

		future_egomotion = torch.Tensor(future_egomotion).float()

		# Convert to 6DoF vector
		future_egomotion = mat2pose_vec(future_egomotion)
		return future_egomotion.unsqueeze(0)

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		"""
		Returns
		-------
			data: dict with the following keys:
				image: torch.Tensor<float> (T, N, 3, H, W)
					normalised cameras images with T the sequence length, and N the number of cameras.
				intrinsics: torch.Tensor<float> (T, N, 3, 3)
					intrinsics containing resizing and cropping parameters.
				extrinsics: torch.Tensor<float> (T, N, 4, 4)
					6 DoF pose from world coordinates to camera coordinates.
				segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
					(H_bev, W_bev) are the pixel dimensions in bird's-eye view.
				instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
				centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
				offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
				flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
				future_egomotion: torch.Tensor<float> (T, 6)
					6 DoF egomotion t -> t+1
				sample_token: List<str> (T,)
				'z_position': list_z_position,
				'attribute': list_attribute_label,

		"""
		data = {}
		keys = ['image', 'intrinsics', 'extrinsics',
				'segmentation', 'instance', 'centerness', 'offset', 'flow', 'future_egomotion',
				'sample_token',
				'z_position', 'attribute'
				]
		for key in keys:
			data[key] = []

		instance_map = {}
		# Loop over all the frames in the sequence.
		for index_t in self.indices[index]:
			rec = self.ixes[index_t]

			images, intrinsics, extrinsics = self.get_input_data(rec)
			segmentation, instance, z_position, instance_map, attribute_label = self.get_label(rec, instance_map)

			future_egomotion = self.get_future_egomotion(rec, index_t)

			data['image'].append(images)
			data['intrinsics'].append(intrinsics)
			data['extrinsics'].append(extrinsics)
			data['segmentation'].append(segmentation)
			data['instance'].append(instance)
			data['future_egomotion'].append(future_egomotion)
			data['sample_token'].append(rec['token'])
			data['z_position'].append(z_position)
			data['attribute'].append(attribute_label)

		for key, value in data.items():
			if key in ['sample_token', 'centerness', 'offset', 'flow']:
				continue
			data[key] = torch.cat(value, dim=0)

		# If lyft need to subsample, and update future_egomotions
		if self.cfg.MODEL.SUBSAMPLE:
			for key, value in data.items():
				if key in ['future_egomotion', 'sample_token', 'centerness', 'offset', 'flow']:
					continue
				data[key] = data[key][::2].clone()
			data['sample_token'] = data['sample_token'][::2]

			# Update future egomotions
			future_egomotions_matrix = pose_vec2mat(data['future_egomotion'])
			future_egomotion_accum = torch.zeros_like(future_egomotions_matrix)
			future_egomotion_accum[:-1] = future_egomotions_matrix[:-1] @ future_egomotions_matrix[1:]
			future_egomotion_accum = mat2pose_vec(future_egomotion_accum)
			data['future_egomotion'] = future_egomotion_accum[::2].clone()

		instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label(
			data['instance'], data['future_egomotion'],
			num_instances=len(instance_map), ignore_index=self.cfg.DATASET.IGNORE_INDEX, subtract_egomotion=True,
			spatial_extent=self.spatial_extent,
		)
		data['centerness'] = instance_centerness
		data['offset'] = instance_offset
		data['flow'] = instance_flow
		return data

# Argoverse Future Prediction Dataset
class ArgoverseFPD(torch.utils.data.Dataset):
	def __init__(self, cfg, mini_set=False, train=False):
		if train:
			self.datapath='data/argoverse-train'
		else:
			self.datapath='data/argoverse-val'

		self.dataset = ArgoverseTrackingLoader(self.datapath)

		self.cfg = cfg
		self.is_train = train

		self.frames_in_past = self.cfg.TIME_RECEPTIVE_FIELD
		self.frames_in_future = self.cfg.N_FUTURE_FRAMES
		assert self.frames_in_future >= 0 and self.frames_in_past >=0

		self.bev_xbound = self.cfg.LIFT.X_BOUND
		self.bev_ybound = self.cfg.LIFT.Y_BOUND
		self.bev_zbound = self.cfg.LIFT.Z_BOUND

		self.image_height = self.dataset.get_image(1, camera=self.dataset.CAMERA_LIST[0]).shape[0]
		self.image_width = self.dataset.get_image(1, camera=self.dataset.CAMERA_LIST[0]).shape[1]
		self.image_size_wh = ( self.image_width, self.image_height)

		self.image_resize_scale = self.cfg.IMAGE.RESIZE_SCALE
		self.image_final_size_wh = (self.cfg.IMAGE.FINAL_DIM[1], self.cfg.IMAGE.FINAL_DIM[0])

		# exclude stereo-cameras
		self.camera_names = self.dataset.CAMERA_LIST[:-2]

		self.sequence_length = self.frames_in_past + self.frames_in_future

		# get tokens for 15-30sec sequence scene
		self.scene_logs = self.dataset.log_list

		# form sequences of frames of length self.sequence_length
		self.frames_sequences = self.__form_frames_sequences()
		
		if mini_set:
			self.frames_sequences = self.frames_sequences[:10]

		# Image resizing and cropping
		self.augmentation_parameters = self.__get_resizing_and_cropping_parameters()

		# Normalising input images
		self.normalise_image = torchvision.transforms.Compose(
			[torchvision.transforms.ToTensor(),
			 torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
		
		self.denormalise_img = torchvision.transforms.Compose(
			(NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		 		torchvision.transforms.ToPILImage(),))

		# Bird's-eye view parameters
		bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
			self.bev_xbound, self.bev_ybound, self.bev_zbound
		)
		self.bev_resolution, self.bev_start_position, self.bev_dimension = (
			bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
		)

		# Spatial extent in bird's-eye view, in meters
		self.spatial_extent = (self.bev_xbound[1], self.bev_ybound[1])
	
	def __form_frames_sequences(self) -> Dict[str, Union[str, int]]:
		sequences = []		
		for log in self.scene_logs:
			self.dataset.current_log = log
			n_images = len(self.dataset.image_list_sync[self.camera_names[0]])
			for idx in range(0, n_images - self.sequence_length * 5, 5):
				sequence = {'log': log, 'frames_ids': [i for i in range(idx, idx + self.sequence_length * 5, 5)]}
				assert sequence['frames_ids'][-1] <= n_images
				sequences.append(sequence)

		return sequences
	
	def __get_resizing_and_cropping_parameters(self) -> Dict[str, Tuple[int]]:

		original_width, original_height = self.image_size_wh
		final_width, final_height = self.image_final_size_wh

		resize_scale = min(original_width//final_width, original_height//final_height)
		resize_scale = 1/resize_scale
		resized_dims = (int(original_width * resize_scale), int(original_height * resize_scale))

		crop_h = int(resized_dims[1] - final_height)
		crop_w = int(max(0, (resized_dims[0] - final_width) / 2))

		# Pixel ids of cropped image in resized image(left, top, right, bottom)
		crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

		return {
				'scale_width': resize_scale,
				'scale_height': resize_scale,
				'resize_dims': resized_dims,
				'crop': crop,
				}

	def __get_data_timestamp(self, log_id: str, frame_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Parameters
		----------
			log: log-string in argoverse dataset (15-30sec sequences are separated into logs)
			frame_id: frame id of image that synchronised with lidar frequency (10hz instead of 30hz)

		Returns
		-------
			N: number of cameras

			images: torch.Tensor<float> (1, N, 3, H, W)
			intrinsics: torch.Tensor<float> (1, N, 3, 3)
			extrinsics: torch.Tensor(1, N, 4, 4)

			Return data from all 7 camera at one timestamp(frame/picture/snapshot)
		"""
		images = []
		intrinsics = []
		extrinsics = []

		for cam in self.camera_names:
			# -------images part---------
			# get image syncronised with lidar (10Hz instead of 30hz) 
			image_filename = self.dataset.get_image_sync(frame_id, camera=cam, log_id=log_id, load=False)
			img = Image.open(image_filename)
			img = resize_and_crop_image(img, 
				resize_dims=self.augmentation_parameters['resize_dims'], 
				crop=self.augmentation_parameters['crop'])
			img = self.normalise_image(img)
			images.append(img.unsqueeze(0).unsqueeze(0))

			
			calibration = self.dataset.get_calibration(camera=cam, log_id=log_id)
			# -------intrinsics part---------
			intrinsics_matrix = torch.from_numpy(calibration.K[0:3, 0:3])
			top_crop = self.augmentation_parameters['crop'][1]
			left_crop = self.augmentation_parameters['crop'][0]
			intrinsics_matrix = update_intrinsics(intrinsics_matrix.float(), top_crop, left_crop,
				scale_width=self.augmentation_parameters['scale_width'],
				scale_height=self.augmentation_parameters['scale_height'])
			intrinsics.append(intrinsics_matrix.unsqueeze(0).unsqueeze(0))
			# -------extrinsics part---------
			T_egovehicle_to_camera = calibration.extrinsic
			T_camera_to_egovehicle = torch.from_numpy(np.linalg.inv(T_egovehicle_to_camera)).float()
			extrinsics.append(T_camera_to_egovehicle.unsqueeze(0).unsqueeze(0))

		images, intrinsics, extrinsics = (torch.cat(images, dim=1),
										  torch.cat(intrinsics, dim=1),
										  torch.cat(extrinsics, dim=1))

		return images, intrinsics, extrinsics
	
	def __get_future_egomotion(self, log_id: str, frame_id: int) -> torch.Tensor:
		"""
		Get relative motion between present egovehicle position in the city(frame) 
		and next egovehicle position in the city.

		Parameters:
			log_id: log id of argoverse dataset log
			frame_id: current frame in the sequence
			last_frame: last frame in the sequence

		Returns:
			future_motion: torch.Tensor<float> (1, 6) -> 
				6 -> Relative egomotion in form of 6DoF (tx, ty, tz, rx, ry, rz)
		"""

		present_T_ego_to_city = self.dataset.get_pose(idx=frame_id, log_id=log_id).transform_matrix
		next_T_ego_to_city = self.dataset.get_pose(idx=frame_id+1, log_id=log_id).transform_matrix
		T_next_to_present = np.linalg.inv(present_T_ego_to_city).dot(next_T_ego_to_city)

		future_egomotion = torch.Tensor(T_next_to_present).float()

		# Convert to 6DoF vector
		future_egomotion = mat2pose_vec(future_egomotion)
		return future_egomotion.unsqueeze(0)

	def __get_label(self, log_id: str, frame_id: int, instance_map: dict):
		segmentation_np, instance_np, z_position_np, instance_map, attribute_label_np = \
			self.__get_birds_eye_view_label(log_id, frame_id, instance_map)
		segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
		instance = torch.from_numpy(instance_np).long().unsqueeze(0)
		z_position = torch.from_numpy(z_position_np).float().unsqueeze(0).unsqueeze(0)
		attribute_label = torch.from_numpy(attribute_label_np).long().unsqueeze(0).unsqueeze(0)

		return segmentation, instance, z_position, instance_map, attribute_label

	def __get_birds_eye_view_label(self, log_id, frame_id, instance_map):
		## Transformation from global frane to egovehicle
		# translation, rotation = self.__get_top_lidar_pose(log_id, frame_id)

		segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
		# Background is ID 0
		instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
		z_position = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
		attribute_label = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

		labeled_objects = self.dataset.get_label_object(frame_id, log_id)
		only_vehicles = [obj for obj in labeled_objects if obj.label_class in ['VEHICLE', 'LARGE_VEHICLE', 'MOPED']]

		for vehicle in only_vehicles:

			# correspond unique vehicle with the color on instance map
			if vehicle.track_id not in instance_map:
				instance_map[vehicle.track_id] = len(instance_map) + 1
			instance_id = instance_map[vehicle.track_id]

			instance_attribute = int(vehicle.occlusion)

			poly_region, z = self.__get_poly_region_in_image(vehicle)
			cv2.fillPoly(instance, [poly_region], instance_id)
			cv2.fillPoly(segmentation, [poly_region], 1.0)
			cv2.fillPoly(z_position, [poly_region], z)
			cv2.fillPoly(attribute_label, [poly_region], instance_attribute)

		return segmentation, instance, z_position, instance_map, attribute_label

	def __get_poly_region_in_image(self, vehicle_object):
		box = vehicle_object.as_3d_bbox()
		bottom_corners = box[[2,3,7,6]]

		# get only x,y of all corners
		pts = bottom_corners[:,:2]
		# convert to the frame of BEV
		pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
		# change x and y to show x-axis of the vehicle to point up on the BEV
		pts[:, [1, 0]] = pts[:, [0, 1]]

		z = bottom_corners[0,2]
		return pts, z

	def __len__(self):
		return len(self.frames_sequences)
	
	def __getitem__(self, index):
		"""
		Returns
		-------
			data: dict with the following keys:
				image: torch.Tensor<float> (T, N, 3, H, W)
					normalised cameras images with T the sequence length, and N the number of cameras.
				intrinsics: torch.Tensor<float> (T, N, 3, 3)
					intrinsics containing resizing and cropping parameters.
				extrinsics: torch.Tensor<float> (T, N, 4, 4)
					6 DoF pose from world coordinates to camera coordinates.
				segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
					(H_bev, W_bev) are the pixel dimensions in bird's-eye view.
				instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
				centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
				offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
				flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
				future_egomotion: torch.Tensor<float> (T, 6)
					6 DoF egomotion t -> t+1
				log_id: str, id of log in argoverse dataset
				'z_position': list_z_position,
				'attribute': list_attribute_label,

		"""
		data = {}
		keys = ['image', 'intrinsics', 'extrinsics', 'future_egomotion',
				'segmentation', 'instance', 'centerness', 'offset', 'flow',
				'log_id',
				'z_position', 'attribute'
				]
		for key in keys:
			data[key] = []

		instance_map = {}
		
		sequence = self.frames_sequences[index]
		log_id = sequence['log']
		frames_ids = sequence['frames_ids']

		data['log_id'] = log_id

		# Loop over all the frames in the sequence.
		for frame_id in frames_ids:
			images, intrinsics, extrinsics = self.__get_data_timestamp(log_id, frame_id)
			segmentation, instance, z_position, instance_map, attribute_label = \
				self.__get_label(log_id, frame_id, instance_map)
			
			# corresponding to frame_id future_egomotion contains 
			# the motion between the frame and the next frame
			if frame_id == frames_ids[-1]:
				future_egomotion = torch.Tensor([0,0,0,0,0,0]).unsqueeze(0)
			else:
				future_egomotion = self.__get_future_egomotion(log_id, frame_id)

			data['image'].append(images)
			data['intrinsics'].append(intrinsics)
			data['extrinsics'].append(extrinsics)
			data['segmentation'].append(segmentation)
			data['instance'].append(instance)
			data['future_egomotion'].append(future_egomotion)
			data['z_position'].append(z_position)
			data['attribute'].append(attribute_label)

		for key, value in data.items():
			if key in ['log_id', 'centerness', 'offset', 'flow']:
				continue
			data[key] = torch.cat(value, dim=0)

		instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label(
			data['instance'], data['future_egomotion'],
			num_instances=len(instance_map), ignore_index=self.cfg.DATASET.IGNORE_INDEX, subtract_egomotion=True,
			spatial_extent=self.spatial_extent,
		)
		data['centerness'] = instance_centerness
		data['offset'] = instance_offset
		data['flow'] = instance_flow
		return data
	

	def visualize_sample(self, data_sample):
		
		#configure plot
		val_w = 2.99
		cameras = self.camera_names
		image_ratio = self.image_final_size_wh[1] / self.image_final_size_wh[0]
		val_h = val_w * image_ratio
		fig = plt.figure(figsize=(5 * val_w, 2 * val_h))
		width_ratios = (val_w, val_w, val_w, val_w, val_w)
		gs = mpl.gridspec.GridSpec(2, 5, width_ratios=width_ratios)
		gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

		# plot images
		images = data_sample['image']
		for img_id, img in enumerate(images[0, :]):
			ax = plt.subplot(gs[img_id // 4, img_id % 4])
			showimg = self.denormalise_img(img.cpu())
			plt.annotate(cameras[img_id], (0.01, 0.87), c='white',
						xycoords='axes fraction', fontsize=14)
			plt.imshow(showimg)
			plt.axis('off')
			# ax.set_title(self.camera_names[img_id])
		
		# plot segmentation image
		instance_image = data_sample['instance']
		unique_ids = torch.unique(instance_image[0]).cpu().long().numpy()[1:]
		instance_map = dict(zip(unique_ids, unique_ids))
		segmentation_img = plot_instance_map(instance_image[0].cpu().numpy(), instance_map)
		# segmentation_img = segmentation_img.reshape((segmentation_img.shape[-3], segmentation_img.shape[-2]))

		ax = plt.subplot(gs[:,-1])
		
		showimg = Image.fromarray(segmentation_img[::-1,::-1])
		plt.imshow(showimg)
		plt.axis('off')
		
		plt.draw()
		plt.waitforbuttonpress()
		plt.close()
		


def prepare_dataloaders(cfg, return_dataset=False):
	version = cfg.DATASET.VERSION
	train_on_training_data = True

	if cfg.DATASET.NAME == 'nuscenes':
		# 28130 train and 6019 val
		dataroot = os.path.join(cfg.DATASET.DATAROOT, version)
		dataset = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
	elif cfg.DATASET.NAME == 'lyft':
		# train contains 22680 samples
		# we split in 16506 6174
		dataroot = os.path.join(cfg.DATASET.DATAROOT, 'trainval')
		dataset = LyftDataset(data_path=dataroot,
						   json_path=os.path.join(dataroot, 'train_data'),
						   verbose=True)

	traindata = FuturePredictionDataset(dataset, train_on_training_data, cfg)
	valdata = FuturePredictionDataset(dataset, False, cfg)

	if cfg.DATASET.VERSION == 'mini':
		traindata.indices = traindata.indices[:10]
		valdata.indices = valdata.indices[:10]

	nworkers = cfg.N_WORKERS
	trainloader = torch.utils.data.DataLoader(
		traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
	)
	valloader = torch.utils.data.DataLoader(
		valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)

	if return_dataset:
		return trainloader, valloader, traindata, valdata
	else:
		return trainloader, valloader

def prepare_argoverse(cfg, mini_version=False, return_dataset=False):


	if mini_version:
		traindata = ArgoverseFPD(cfg, mini_version, train=True)
		valdata = ArgoverseFPD(cfg, mini_version, train=False)
	else:
		traindata = ArgoverseFPD(cfg, train=True)
		valdata = ArgoverseFPD(cfg, train=False)

	nworkers = cfg.N_WORKERS
	trainloader = torch.utils.data.DataLoader(
		traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
	)
	valloader = torch.utils.data.DataLoader(
		valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)

	if return_dataset:
		return trainloader, valloader, traindata, valdata
	else:
		return trainloader, valloader


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Fiery training')
	parser.add_argument('--config-file', 
		default='fiery/configs/argoverse/baseline.yml', metavar='FILE', help='path to config file')
	parser.add_argument('--nuscenes', default=False, help='True enables NuScenes dataset to debug on')
	parser.add_argument('--mini', default=False, help='True enables training on mini dataset')
	args = parser.parse_args()
	
	if args.nuscenes:
		cfg = get_cfg()
		dataroot = os.path.join(cfg.DATASET.DATAROOT, cfg.DATASET.VERSION)
		dataset = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
		traindata = FuturePredictionDataset(dataset, is_train=True, cfg=cfg)
	else:
		cfg = get_cfg(args)

	argoverse = ArgoverseFPD(cfg)
	something = argoverse.visualize_sample(argoverse[9])