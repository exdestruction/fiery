import os
import time
import socket
import torch
import argparse

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from fiery.config import get_parser, get_cfg
from fiery.data import prepare_dataloaders, prepare_argoverse
from fiery.trainer import TrainingModule


def main(args):
	torch.cuda.empty_cache()
	# cfg = get_cfg()
	cfg = get_cfg(args)

	if cfg.DATASET.NAME == 'argoverse':
		trainloader, valloader = prepare_argoverse(cfg, mini_version=False)
	else:
		trainloader, valloader = prepare_dataloaders(cfg)

	model = TrainingModule(cfg.convert_to_dict())

	if cfg.PRETRAINED.LOAD_WEIGHTS:
		# Load single-image instance segmentation model.
		pretrained_model_weights = torch.load(
			os.path.join(cfg.DATASET.DATAROOT, cfg.PRETRAINED.PATH), map_location='cpu'
		)['state_dict']

		model.load_state_dict(pretrained_model_weights, strict=False)
		print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

	save_dir = os.path.join(
		cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
	)

	

	tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
	trainer = pl.Trainer(
		gpus=cfg.GPUS,
		accelerator='ddp',
		precision=cfg.PRECISION,
		sync_batchnorm=True,
		gradient_clip_val=cfg.GRAD_NORM_CLIP,
		max_epochs=cfg.EPOCHS,
		weights_summary='full',
		logger=tb_logger,
		log_every_n_steps=cfg.LOGGING_INTERVAL,
		plugins=DDPPlugin(find_unused_parameters=True),
		profiler='simple',
	)
	trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Fiery training')
	parser.add_argument('--config-file', default='fiery/configs/argoverse/baseline.yml', metavar='FILE', help='path to config file')
	parser.add_argument(
		'opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER,
	)
	args = parser.parse_args()
	main(args)
