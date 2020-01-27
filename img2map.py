# @Date:   2018-08-01T11:06:48+02:00
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2020-01-27T15:18:56+01:00

import sys
from img2mapC import img2mapC

import numpy as np
from keras.models import load_model
import h5py
import os
import glob2
import scipy.io as sio
from scipy import stats

import modelSelection
from keras.optimizers import Nadam
nadam = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)

########################################################
modelPath='./data/'
fileD='./data/img/'

nnList = [12]
########################################################
MapfileD=modelPath+'pre/'
if not os.path.exists(MapfileD):
	os.makedirs(MapfileD)
########################################################
patch_shape = (128, 128, 10)
step=patch_shape[0]
params = {'dim_x': patch_shape[0],
		   'dim_y': patch_shape[1],
		   'dim_z': patch_shape[2],
		   'step': step,
		   'Bands': [1,2,3,4,5,6,7,8,11,12],
		   'scale':10000.0,
		   'nanValu':999}

img2mapCLass=img2mapC(**params);

for nn in nnList:#

	model, dataFlow = modelSelection.modelSelection(nn, nadam, patch_shape=patch_shape)
	modelName =  modelPath + '_'+ str(nn)+"_8weights.best.hdf5"

	print(modelName)
	model.load_weights(modelName, by_name=False)

	files=glob2.glob(fileD+'*.tif')

	for file in files:

			filename=os.path.basename(file);
			mapFile = MapfileD+filename[:-4]+'_'+ str(nn)

			print(params['Bands'])
			print(files, mapFile)
			img2mapCLass.img2Bdetection_ovlp([file], model, mapFile, nn=nn)
