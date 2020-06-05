# @Date:   2019-05-13
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2020-06-05T20:57:30+02:00


import sys
from dataGener import DataGenerator

import dataPre
import modelSelection

import random
from pathlib import Path
import h5py
import numpy as np
import pickle
import glob
import time
import scipy.io as sio

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Nadam

import tensorflow as tf
from keras import backend as K
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.49#0.41
session = tf.Session(config=config)

#################################################################folder
file0='./'
nadam = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
nnList = [12]
epochs = 100

###################################################################
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
patch_shape = (128, 128, 10)
batch_size = 8
fileTra, fileVal, traNum, valNum=dataPre.traValFiles_spatial()
###################################################################
if not os.path.exists(file0):
	os.makedirs(file0)

print('nn',nnList)

for nn in nnList:
    timeCreated=time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))

    fileSave=  file0 + '_' + str(nn) + '_' + str(batch_size)
    checkpoint = ModelCheckpoint(fileSave+"weights.best.hdf5", monitor='val_f1_m', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
    # checkpoint_ = ModelCheckpoint(fileSave+'model{epoch:08d}.h5', period=10)

    tbCallBack = TensorBoard(log_dir=file0+'logs' + '_' + str(nn) + '_' + str(batch_size) + '_' + timeCreated,  # log 目录
                     histogram_freq=0,
                     write_graph=True,
                     write_grads=True,
                     write_images=True,
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    model, dataFlow = modelSelection.modelSelection(nn, nadam, patch_shape=patch_shape)

	# Generators
    params = {'dim_x': patch_shape[0],
			  'dim_y': patch_shape[1],
			  'dim_z': patch_shape[2],
			  'batch_size':batch_size,
			  'flow': dataFlow}
    tra_generator = DataGenerator(**params).generate(fileTra)
    val_generator = DataGenerator(**params).generate(fileVal)

    if epochs==1:
    	# 'find the range of feasible lr'
    	# sys.path.insert(0, '/home/qiu/CodeSummary/LCZ_CPQiu_util/lr')
    	# from LRFinder import LRFinder
		#
    	# traNum=100000
    	# valNum=8
    	# lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-3, steps_per_epoch=np.ceil(traNum/batch_size), epochs=epochs, pathS=file0 +str(nn)+'_'+str(batch_size))
		#
    	model.fit_generator(generator = tra_generator,
    				steps_per_epoch = traNum//batch_size, epochs = epochs,
    				validation_data = val_generator,
    				validation_steps = valNum//batch_size,
    				callbacks = [lr_finder, tbCallBack], max_queue_size = 100)#
    	lr_finder.plot_loss()
    	lr_finder.plot_lr()

    else:
    	start = time.time()
    	model.fit_generator(generator = tra_generator,
    					steps_per_epoch = traNum//batch_size, epochs = epochs,
    					validation_data = val_generator,
    					validation_steps = valNum//batch_size,
    					callbacks = [checkpoint, tbCallBack, early_stopping], max_queue_size = 100, verbose=1)
						#, logBatch , early_stopping , clr_triangular checkpoint_,
    	end =time.time()

    	trainingTime=end-start;
    	savedModel = fileSave + 'model.final_' +'.h5'
    	model.save_weights(savedModel)
    	sio.savemat((fileSave+'_trainingTime_.mat'), {'trainingTime':trainingTime})
