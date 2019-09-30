import os
import logging
import argparse
import numpy as np

from train_and_evaluate import evaluate, train
from model.net import Generator, Discriminator
from data_loader import fetch_dataloader
import utils
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='Result',
										help="Result folder")
parser.add_argument('--train_path', default='Data/trainset.nc',
										help="The training dataset path")
parser.add_argument('--restore_from', default=None,
										help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
	# Load the directory from commend line
	args = parser.parse_args()
	train_path = args.train_path
	output_dir = args.output_dir
	restore_from = args.restore_from

	os.makedirs(output_dir + '/outputs', exist_ok = True)
	os.makedirs(output_dir + '/figures', exist_ok = True)
	os.makedirs(output_dir + '/model', exist_ok = True)

	 # Set the logger
	utils.set_logger(os.path.join(args.output_dir, 'train.log'))

	if restore_from is None:
		restore = 0
	else:
		restore = 1


	# Load parameters from json file
	json_path = os.path.join(args.output_dir,'Params.json')
	assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
	params = utils.Params(json_path)

	# Add attributes to params
	params.output_dir = output_dir
	params.lambda_gp  = 10.0
	params.n_critic = 1
	params.cuda = torch.cuda.is_available()
	
	params.batch_size = int(params.batch_size)
	params.numIter = int(params.numIter)
	params.noise_dims = int(params.noise_dims)
	params.label_dims = int(params.label_dims)
	params.gkernlen = int(params.gkernlen)
	

	# fetch dataloader
	dataloader = fetch_dataloader(train_path, params)

	# Define the models 
	generator = Generator(params)
	discriminator = Discriminator(params)
	if params.cuda:
		generator.cuda()
		discriminator.cuda()


	# Define the optimizers 
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=params.lr_gen, betas=(params.beta1_gen, params.beta2_gen))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=params.lr_dis, betas=(params.beta1_dis, params.beta2_dis))

	# train the model and save 
	logging.info('Start training')
	loss_history = train((generator, discriminator), (optimizer_G, optimizer_D), dataloader, params)

	# plot loss history and save
	utils.plot_loss_history(loss_history, output_dir)

	# Generate images and save 
	wavelengths = [w for w in range(500, 1301, 50)]
	angles = [a for a in range(35, 86, 5)]

	logging.info('Start generating devices for wavelength range {} to {} and angle range from {} to {} \n'
				.format(min(wavelengths), max(wavelengths), min(angles), max(angles)))
	evaluate(generator, wavelengths, angles, num_imgs=500, params=params)




