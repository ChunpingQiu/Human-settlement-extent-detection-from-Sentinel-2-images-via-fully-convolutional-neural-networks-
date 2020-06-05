
import glob
import numpy as np
import scipy.io as sio

def traValFiles_random():

	cities = ['LCZ42_204296_Berlin',
			  'LCZ42_22167_Lisbon',
			  'LCZ42_22549_Madrid',
			  'LCZ42_21571_Milan',
			  'LCZ42_20985_Paris']
	folderData = '/work/qiu/data4mt/data/'
	seasonNum = 3

	#number of training and validate samples
	mat = sio.loadmat(folderData+'vali/'+'patchNum.mat')
	patchNum=mat['patchNum']*1
	valNum_=np.sum(patchNum[0,:]) * seasonNum ; #

	mat = sio.loadmat(folderData+'trai/'+'patchNum.mat')
	patchNum=mat['patchNum']*1
	traNum_=np.sum(patchNum[0,:]) * seasonNum ;

	fileVal=[]
	#validata
	fileTra=[]
	#validata samples

	for city in cities:
		for id in np.arange(25):
			fileVal.append(folderData+'trai/' +city +'_'+str(id)+'.h5')
			fileVal.append(folderData+'trai/' +city +'_spring_'+str(id)+'.h5')
			fileVal.append(folderData+'trai/' +city +'_autumn_'+str(id)+'.h5')

			fileVal.append(folderData+'vali/' +city +'_'+str(id)+'.h5')
			fileVal.append(folderData+'vali/' +city +'_spring_'+str(id)+'.h5')
			fileVal.append(folderData+'vali/' +city +'_autumn_'+str(id)+'.h5')

		for id in np.arange(25,100):

			fileTra.append(folderData+'trai/' +city +'_'+str(id)+'.h5')
			fileTra.append(folderData+'trai/' +city +'_spring_'+str(id)+'.h5')
			fileTra.append(folderData+'trai/' +city +'_autumn_'+str(id)+'.h5')

			fileTra.append(folderData+'vali/' +city +'_'+str(id)+'.h5')
			fileTra.append(folderData+'vali/' +city +'_spring_'+str(id)+'.h5')
			fileTra.append(folderData+'vali/' +city +'_autumn_'+str(id)+'.h5')

	print('train files:', len(fileTra))
	print('vali files:',  len(fileVal))

	traNum=np.int32((traNum_+valNum_)*0.75)
	valNum=np.int32((traNum_+valNum_)*0.25)

	return fileTra, fileVal, traNum, valNum

def traValFiles_spatial():

	folderData = '/work/qiu/data4mt/data/'
	seasonNum = 3

	#number of training and validate samples
	mat = sio.loadmat(folderData+'vali/'+'patchNum.mat')
	patchNum=mat['patchNum']*1
	valNum=np.sum(patchNum[0,:]) * seasonNum ; #

	mat = sio.loadmat(folderData+'trai/'+'patchNum.mat')
	patchNum=mat['patchNum']*1
	traNum=np.sum(patchNum[0,:]) * seasonNum ;

	#validata samples
	fileVal = glob.glob(folderData+'vali/' +'*.h5')#、glob2.
	#training samples
	fileTra = glob.glob(folderData+'trai/' +'*.h5')
	print('train files:', len(fileTra))
	print('vali files:',  len(fileVal))

	return fileTra, fileVal, traNum, valNum
