# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 21:06:42 2017

@author: qiu
"""

import numpy as np
import h5py
from random import shuffle
import random
import glob
import re


class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x, dim_y, dim_z, batch_size, flow):
	  'Initialization'
	  self.dim_x = dim_x
	  self.dim_y = dim_y
	  self.dim_z = dim_z
	  self.batch_size = batch_size
	  self.flow = flow

  def generate(self, fileList):
	  random.seed(4)

	  # random the order the files
	  #'Generates batches of samples'
	  # Infinite loop
	  while 1:
		  # Generate batches
		  shuffle(fileList)
		  #print(fileList)

		  #for every file in the train/test folder
		  for fileD in fileList:

			  #load all the data in this file
			  hf = h5py.File(fileD, 'r')

			  #print(keyNameX)
			  x_thisFile=np.array(hf.get('x'))
			  if self.flow!=0:
					   y_thisFile_=np.array(hf.get('y_1'))
			  if self.flow!=1:
					   y_thisFile=np.array(hf.get('y_0'))
			  hf.close()

			  nb_thisFile = np.shape(x_thisFile)
			  #print(nb_thisFile)

			  #how many batch_size in this file:      imax
			  imax = int(nb_thisFile[0]/self.batch_size)
			  #print(imax)

			   #the data in each file have already in a random order, but this is to make sure the data order in each epoch is different
			  indexes = np.arange(nb_thisFile[0], dtype=np.uint32)
			  np.random.shuffle(indexes)
			  for i in range(imax):
					 # Find list of IDs in this loop
					 ID = indexes[i*self.batch_size:(i+1)*self.batch_size]

					 # init
					 X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
					 X = x_thisFile[ID]

					 if self.flow!=0:
						   #y_1 = np.empty((self.batch_size, self.dim_x, self.dim_y, np.shape(y_thisFile_)[-1]), dtype = int)
						   y_1 = y_thisFile_[ID]
					 if self.flow!=1:
						   y_0 = y_thisFile[ID]
						   # 'because tf.losses.sparse_softmax_cross_entropy is used'
						   # y_0 = np.expand_dims(y_thisFile[ID,:,:,0], axis=-1)

					 if self.flow==2:
						   yield X, [y_0/100.0, y_1]
					 if self.flow==0:
						   yield X, y_0/100.0
