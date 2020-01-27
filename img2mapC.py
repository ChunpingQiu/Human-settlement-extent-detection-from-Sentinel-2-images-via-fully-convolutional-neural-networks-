# @Date:   2018-07-16T20:51:18+02:00
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2020-01-27T14:55:47+01:00


import sys
import os
import numpy as np
from skimage.util.shape import view_as_windows
import glob
from osgeo import gdal,osr
import glob2
from scipy import stats
import scipy.ndimage
from memprof import *
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.models import Model
from keras.layers import Input

'load img tif and get patches from it;  save as tif file after getting predictions'
class img2mapC(object):

  def __init__(self, dim_x, dim_y, dim_z, step, Bands, scale, nanValu):
	  self.dim_x = dim_x#shape of the patch
	  self.dim_y = dim_y
	  self.dim_z = dim_z
	  self.step = step#
	  self.Bands = Bands#bands selected from the image files, list
	  self.scale = scale#the number used to divided the pixel value by
	  self.isSeg = 1#whether segmentation
	  self.nanValu = nanValu

  '''
    # cut a matrix into patches
    # input:
            imgMat: matrix containing the bands of the image
    # output:
            patch: the patches, one patch is with the shape: dim_x, dim_y, dim_z
            R: the size of the final lcz map
            C: the size of the final lcz map
            idxNan: the index of no data area.
  '''
  def Bands2patches(self, imgMat, upSampleR):

	  print('imgMat', imgMat.shape, imgMat.dtype)
	  for band in np.arange(imgMat.shape[2]):

		  arr = imgMat[:,:,band]
		  if upSampleR!=1:
		            arr=scipy.ndimage.zoom(arr, [upSampleR,  upSampleR], order=1)#Bilinear interpolation would be order=1

		  patch0, R, C= self.__img2patch(arr)#'from band to patches'
		  #print('patch0', patch0.shape, patch0.dtype)


		  if band==0:

			#find the nodata area
			  patch0Tmp=np.amin(patch0, axis=1);
			#print(b) # axis=1；每行的最小值
			  indica=np.amin(patch0Tmp, axis=1);

			  idxNan = np.where( (indica<0.000001) )
			  idxNan = idxNan[0].reshape((-1,1))
			  print(idxNan.shape)

			  patch=np.zeros(((patch0.shape[0]-idxNan.shape[0]), self.dim_x, self.dim_y, imgMat.shape[2]), dtype=imgMat.dtype);

		  patch0 = np.delete(patch0, idxNan, axis=0)
		  if self.scale == -1:#scale with a fucntion
			  patch[:,:,:,band]=self.scaleBand(patch0)

		  else:
			  patch[:,:,:,band]=patch0/self.scale ;

	  return patch, R, C, idxNan


# #from a multi bands mat to patches, without considering the nan area
  def Bands2patches_all(self, imgMat, upSampleR):
	  for band in np.arange(imgMat.shape[2]):

		  arr = imgMat[:,:,band]
		  if upSampleR!=1:
		            arr=scipy.ndimage.zoom(arr, [upSampleR,  upSampleR], order=1)#Bilinear interpolation would be order=1
		  patch0, R, C= self.__img2patch(arr)#'from band to patches'

		  if band==0:
			  patch=np.zeros(((patch0.shape[0]), self.dim_x, self.dim_y, imgMat.shape[2]), dtype=imgMat.dtype);

		  #print('self.scale', self.scale)
		  if self.scale == -1:#scale with a fucntion
			  patch[:,:,:,band]=self.scaleBand(patch0)

		  else:
			  patch[:,:,:,band]=patch0/self.scale ;

	  return patch, R, C


  '''
    # load all relevent bands of a image file
    # input:
            imgFile: image file
    # output:
            prj: projection data
            trans: projection data
            matImg: matrix containing the bands of the image
  '''
  def loadImgMat(self, imgFile):
	  src_ds = gdal.Open( imgFile )

	  if src_ds is None:
		  print('Unable to open INPUT.tif')
		  sys.exit(1)
	  prj=src_ds.GetProjection()
	  trans=src_ds.GetGeoTransform()
	  #print("[ RASTER BAND COUNT ]: ", src_ds.RasterCount)
	  #print(prj)
	  #print(trans)
	  #print(self.Bands)

	  bandInd=0
	  print(self.Bands)
	  for band in self.Bands:
		  band += 1
		  srcband = src_ds.GetRasterBand(band)

		  if srcband is None:
			  print('srcband is None'+str(band)+imgFile)
			  continue

		  #print('srcband read:'+str(band))

		  arr = srcband.ReadAsArray()
		  #print(np.unique(arr))

		  if bandInd==0:
			  R=arr.shape[0]
			  C=arr.shape[1]
			  #print(arr.shape)
			  matImg=np.zeros((R, C, len(self.Bands)), dtype=np.float32);
		  matImg[:,:,bandInd]=np.float32(arr)#/self.scale ;

		  bandInd += 1

	  return prj, trans, matImg


  'use the imgmatrix to get patches'
# input:
#        mat: a band of the image
# output:
		# patch: the patch of this input image, to feed to the classifier
		# R: the size of the final lcz map
		# C: the size of the final lcz map
  def  __img2patch(self, mat):

      #mat=np.pad(mat, ((np.int(self.dim_x_img/2), np.int(self.dim_x_img/2)), (np.int(self.dim_y_img/2), np.int(self.dim_y_img/2))), 'reflect')

      window_shape = (self.dim_x, self.dim_y)#self.dim_x_img

	  #window_shape = (self.dim_x, self.dim_y)#self.dim_x_img
      B = view_as_windows(mat, window_shape, self.step)#B = view_as_windows(A, window_shape,2)
      #print(B.shape)

      patches=np.reshape(B, (-1, window_shape[0], window_shape[1]))
      #print(patches.shape)

      R=B.shape[0]#the size of the final map
      C=B.shape[1]

      return patches, R, C


  '''
    # save a map as tif
    # input:
            mat: the matrix to be saved
            prj: projection data
            trans: projection data
            mapFile: the file to save the produced map
    # output:
            no
  '''
  def predic2tif(self, mat, prj, trans, mapFile):

	  R=mat.shape[0]
	  C=mat.shape[1]

	  totalNum=R*C;

	  if self.isSeg==1:
		  xres =trans[1]*1
		  yres= trans[5]*1
		  geotransform = (trans[0]+xres*(1-1)/2.0, xres, 0, trans[3]+xres*(1-1)/2.0, 0, yres)
		  print(geotransform)
	  else:
		  xres =trans[1]*self.step
		  yres= trans[5]*self.step
		  #geotransform = (trans[0]+xres*(self.step-1)/2.0, xres, 0, trans[3]+xres*(self.step-1)/2.0, 0, yres)

		  #geotransform = (trans[0] + trans[1]*(2-1)/2.0 -xres*(2-1)/2.0, xres, 0, trans[3] + trans[5]*(2-1)/2.0-yres*(2-1)/2.0, 0, yres)#no padding
		  geotransform = (trans[0] + trans[1]*(self.dim_x-1)/2.0, xres, 0, trans[3] + trans[5]*(self.dim_y-1)/2.0, 0, yres)#no padding

		  #geotransform = (trans[0]-xres*(2-1)/2.0, xres, 0, trans[3]-yres*(2-1)/2.0, 0, yres)#with padding

		  print(geotransform)

	  dimZ=mat.shape[2]

	 # create the dimZ raster file
	  dst_ds = gdal.GetDriverByName('GTiff').Create(mapFile, C, R, dimZ, gdal.GDT_UInt16)#gdal.GDT_Byte .GDT_Float32 gdal.GDT_Float32
	  dst_ds.SetGeoTransform(geotransform) # specify coords
	  dst_ds.SetProjection(prj)

	  for i in np.arange(dimZ):
		  map=mat[:,:,i]
		  dst_ds.GetRasterBand(int(i+1)).WriteArray(map)   # write band to the raster
	  dst_ds.FlushCache()                     # write to disk
	  dst_ds = None


  def scaleBand(self,patches):
      patches_=np.zeros(patches.shape, dtype=np.float32)
      #for b in np.arange(patches.shape[-1]):

      patch=patches.reshape(-1,1)
        #print(patch.shape)
      scaler = StandardScaler().fit(patch)
        #print(scaler.mean_.shape)
      patches_=scaler.transform(patch).reshape(patches.shape[0],patches.shape[1], patches.shape[2])

      return patches_

  'prediction of an image, averaged by four predictions with overlap'
  def img2Bdetection_ovlp(self, file, model, mapFile, out=1, nn=0):

      prj, trans, img= self.loadImgMat(file[0])
      R=img.shape[0]
      C=img.shape[1]
      print('img:', R, C)

      imgN=len(file);

      if self.dim_x==32:
          paddList=[0,8,16,24]
      if self.dim_x==64:
          paddList=[0,16,32,48]
      else:
          paddList=[0,32,64,96]


      for padding in paddList:

          #prepare x_test
          #if imgN==1:
          if padding==0:
              img1=img
          else:
              img1=np.pad(img, ((padding, 0), (padding, 0), (0,0)), 'reflect')
          print(img1.shape)
          x_test, mapR, mapC = self.Bands2patches_all(img1,1)
          print('x_test:', x_test.shape)

          if out==1:

              if nn==18 or nn==16:
                  y = model.predict([x_test[:,:,:,0],x_test[:,:,:,1],x_test[:,:,:,2],x_test[:,:,:,3],x_test[:,:,:,4],x_test[:,:,:,5],x_test[:,:,:,6],x_test[:,:,:,7],x_test[:,:,:,8],x_test[:,:,:,9]], batch_size = 16, verbose=1)
              elif  nn==24 or nn==25:#[1,2,3,7]], X[:,:,:,[0,4,5,6,8,9]
                  y = model.predict([x_test[:,:,:,[1,2,3,7]],x_test[:,:,:,[0,4,5,6,8,9]]])
              else:
                  y = model.predict(x_test, batch_size = 16, verbose=1)

              C, mapPatch_shape =self.pro_from_x(mapR, mapC, y, padding)
              OS = np.int( self.dim_x/ mapPatch_shape )   #ratio between the input and the output
              if padding==0:
                  r=C.shape[0]
                  c=C.shape[1]
                  Pro=C[0:(r-mapPatch_shape),0:(c-mapPatch_shape),:]
              else:
                  Pro=Pro+C[np.int(padding/OS):(r-mapPatch_shape+np.int(padding/OS)), np.int(padding/OS):(c-mapPatch_shape+np.int(padding/OS)), :]

      if out==1:
          self.save_pre_pro(prj, trans, Pro, mapFile, mapPatch_shape)
          return prj, trans, Pro, mapPatch_shape


  ''' get y of the targeting shape'''
  def pro_from_x(self, mapR, mapC, y, padding):

      mapPatch_shape=y.shape[1]
      print('class num:', y.shape[-1])

      B_=np.reshape(y, (mapR, mapC, y.shape[1], y.shape[2], y.shape[-1]))
      print('B_.shape', B_.shape)
      del y

      C=np.zeros((B_.shape[0]*B_.shape[2], B_.shape[1]*B_.shape[3], B_.shape[4]), dtype=float)
      for dim in np.arange(B_.shape[4]):
          B_1=B_[:,:,:,:,dim]
          C[:,:,dim]=B_1.transpose(0,2,1,3).reshape(-1,B_1.shape[1]*B_1.shape[3])
          del B_1
      return C, mapPatch_shape

  ''' save predictions and pro'''
  def save_pre_pro(self, prj, trans, Pro, mapFile, mapPatch_shape):

      y=Pro.argmax(axis=2)+1
      #y[y== 2] = 0#self.nanValu#to make sure only the buildings are labeled

      mapConfi=np.zeros((y.shape[0], y.shape[1], 1), dtype=np.uint16)
      mapConfi[:,:,0]=y;

      mapPro=np.zeros((y.shape[0], y.shape[1], 1), dtype=np.uint16)
      mapPro= Pro*10000;

      #if mapPatch_shape*2==self.dim_x:
      ratio=self.dim_x / mapPatch_shape;
      print('downsampling by: ', ratio)
      trans0 =trans[0]+trans[1]*(ratio-1)/2.0
      trans3= trans[3]+trans[5]*(ratio-1)/2.0
      trans1 =trans[1]* ratio
      trans5= trans[5]* ratio
      trans = (trans0, trans1, 0, trans3, 0, trans5)

      self.predic2tif(mapConfi, prj, trans, mapFile+'.tif')
      self.predic2tif(mapPro, prj, trans, mapFile+'_pro.tif')

##################################################################################
  # ''' generate class prediction from the input samples'''
  # def predict_classes(self, x):
	#   y=x.argmax(axis=1)+1
	#   return y

# '''
#   # save prediction as tif
#   # input:
#           yPre0: the vector of the predictions
#           R: the size of the final lcz map
#           C: the size of the final lcz map
#           prj: projection data
#           trans: projection data
#           mapFile: the file to save the produced map
#           idxNan: the index of no data area.
#   # output:
#           no
# '''
# def predic2tif_vector(self, yPre0, R, C, prj, trans, mapFile, idxNan):
  #   totalNum=R*C;
#
  #   if self.isSeg==1:
  # 	  xres =trans[1]*1
  # 	  yres= trans[5]*1
  # 	  geotransform = (trans[0]+xres*(1-1)/2.0, xres, 0, trans[3]+xres*(1-1)/2.0, 0, yres)
  #   else:
  # 	  xres =trans[1]*self.step
  # 	  yres= trans[5]*self.step
#         #geotransform = (trans[0]+xres*(self.self.dim_x_img-1)/2.0, xres, 0, trans[3]+yres*(self.dim_y_img-1)/2.0, 0, yres)#no padding
  # 	  geotransform = (trans[0]-xres*(2-1)/2.0, xres, 0, trans[3]-yres*(2-1)/2.0, 0, yres)#with padding
#
  #   dimZ=np.shape(yPre0)[1]
#
  #  # create the dimZ raster file
  #   dst_ds = gdal.GetDriverByName('GTiff').Create(mapFile, C, R, dimZ, gdal.GDT_UInt16)#gdal.GDT_Byte .GDT_Float32 gdal.GDT_Float32
  #   dst_ds.SetGeoTransform(geotransform) # specify coords
  #   dst_ds.SetProjection(prj)
#
  #   for i in np.arange(dimZ):
  # 	  yPre=np.zeros((totalNum,1), dtype=np.uint16 ) + self.nanValu + 1;
  # 	  yPre[idxNan]= self.nanValu;# set no data value
  # 	  tmp = np.where( (yPre== self.nanValu + 1 ) )
#
  # 	  yPre[ tmp[0]]=yPre0[:,i].reshape((-1,1));
#
  # 	  map=np.reshape(yPre, (R, C))
  # 	  dst_ds.GetRasterBand(int(i+1)).WriteArray(map)   # write band to the raster
  #   dst_ds.FlushCache()                     # write to disk
  #   dst_ds = None

  #
  # '''
  #    # create a list of files dir for all the cities needed to be produced
  #    # input:
  #            fileD: cities path
  #            cities: a list of cities under the fileD
  #    # output:
  #            files: all the files
  #            imgNum_city: the image number of each city
  # '''
  # def createFileList_cities(self, fileD, cities):
	#   files = []
	#   imgNum_city = np.zeros((len(cities),1), dtype=np.uint8)
	#   for j in np.arange(len(cities)):
	# 		 #all seasons
	# 		 file = sorted(glob2.glob(fileD+ cities[j] +'/**/*_'  + '*.tif'))
	# 		 files.extend(file)
	# 		 imgNum_city[j] = len(file)
  #
	#   return files, imgNum_city
  #
  # '''
  #     # create a list of files dir for all the images in different seasons of the input city dir
  #     # input:
  #             fileD: the absolute path of one city
  #     # output:
  #             files: all the files corresponding to different seasons
  # '''
  # def createFileList(self, fileD):
	#   files = []
	#   imgNum_city = np.zeros((1,1), dtype=np.uint8)
  #
	#  #all seasons
	#   file = sorted(glob2.glob(fileD +'/**/*_'  + '*.tif'))
	#   files.extend(file)
	#   return files
