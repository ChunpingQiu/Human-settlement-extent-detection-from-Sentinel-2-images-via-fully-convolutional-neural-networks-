# @Date:   2019-12-18T16:30:15+01:00
# @Last modified time: 2020-01-27T14:57:11+01:00

import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras import backend as keras
import tensorflow as tf

def sen2IS_net_bn_core(inputs, dim=16, inc_rate=2, bn=1):
    conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(inputs)
    if bn==1:
        print('with BN')
        conv0 = BatchNormalization(axis=-1)(conv0)
    conv0 = Activation('relu')(conv0)
    conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv0)
    if bn==1:
        print('with BN')
        conv0 = BatchNormalization(axis=-1)(conv0)
    conv0 = Activation('relu')(conv0)

    dim=dim*inc_rate
    conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv0)
    if bn==1:
        print('with BN')
        conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv1)
    if bn==1:
        print('with BN')
        conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D((2, 2))(conv1)
    pool2 = AveragePooling2D((2, 2))(conv1)
    merge1 = Concatenate()([pool1,pool2])

    return merge1

def sen2IS_net_bn_core_2(merge1, dim=128, inc_rate=2, numC=2, bn=1, attentionS=0,  attentionC=0):

    conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge1)
    if bn==1:
        print('with BN')
        conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv2)
    if bn==1:
        print('with BN')
        conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)
    drop0 = Dropout(0.1)(conv2)

    dim=dim*inc_rate
    conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(drop0)
    if bn==1:
        print('with BN')
        conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv3)
    if bn==1:
        print('with BN')
        conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)
    drop1 = Dropout(0.1)(conv3)

    'attention part'
    # if  (attentionS==1 and  attentionC==1):
	#        o_c= attention.CAM(drop1, 64)
	#        o_s = attention.PAM(drop1, 64)
	#        drop1 = Add()([o_c, o_s])
    #
    # if  (attentionS==1 and  attentionC==0):
	#        drop1 = attention.PAM(drop1, 64)
    #
    # if  (attentionS==0 and  attentionC==1):
	#        drop1 = attention.CAM(drop1, 64)

    o = Conv2D(numC, (1, 1), padding='same', activation = 'softmax')(drop1)


    return o
##########################################################################################
def sen2IS_net_bn(input_size = (128,128,10), numC=2, ifBN=1, attentionS=0,  attentionC=0):

    inputs = Input(input_size)
    merge1 = sen2IS_net_bn_core(inputs, bn=ifBN)
    o = sen2IS_net_bn_core_2(merge1, dim=128, inc_rate=2, numC=numC, bn=ifBN, attentionS=attentionS,  attentionC=attentionC)

    model = Model(input = inputs, output = o)

    return model


def sen2IS_net_wide(input_size = (128,128,10), numC=2, ifBN=0):

    inputs = Input(input_size)
    merge1 = sen2IS_net_bn_core(inputs, dim=32, inc_rate=2, bn=ifBN)
    o = sen2IS_net_bn_core_2(merge1, dim=256, inc_rate=2, numC=numC, bn=ifBN)

    model = Model(input = inputs, output = o)

    return model
def sen2IS_net_deep(input_size = (128,128,10), numC=2, depth=17):

    inputs = Input(input_size)
    lay_per_block=int((depth-1)/4)

    inc_rate=2
    dim=16

    conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(inputs)
    conv0 = Activation('relu')(conv0)
    for i in np.arange(lay_per_block-1):
        print(i)
        conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv0)
        conv0 = Activation('relu')(conv0)


    dim=dim*inc_rate
    conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv0)
    conv1 = Activation('relu')(conv1)
    for i in np.arange(lay_per_block-1):
        print(i)
        conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv1)
        conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D((2, 2))(conv1)
    pool2 = AveragePooling2D((2, 2))(conv1)
    merge1 = Concatenate()([pool1,pool2])

    ##############################################################
    dim=dim*inc_rate*inc_rate
    conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge1)
    conv2 = Activation('relu')(conv2)
    for i in np.arange(lay_per_block-1):
        print(i)
        conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv2)
        conv2 = Activation('relu')(conv2)
    drop0 = Dropout(0.1)(conv2)

    dim=dim*inc_rate
    conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(drop0)
    conv3 = Activation('relu')(conv3)
    for i in np.arange(lay_per_block-1):
        print(i)
        conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv3)
        conv3 = Activation('relu')(conv3)
    drop1 = Dropout(0.1)(conv3)

    o = Conv2D(numC, (1, 1), padding='same', activation = 'softmax')(drop1)

    model = Model(input = inputs, output = o)

    return model


##########################################################################################
'baselines'
def unet(pretrained_weights = None, input_size = (128,128,10), numC=2, attentionS=0, attentionC=0):
    inputs = Input(input_size)

    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    'end of the downsampling'

    drop5=UpSampling2D(size = (2,2))(drop5)
    up6 = Conv2D(512, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)

    merge6 = Concatenate()([drop4,up6])
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    conv6=UpSampling2D(size = (2,2))(conv6)
    up7 = Conv2D(256, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    merge7 = Concatenate()([conv3,up7])
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    conv7=UpSampling2D(size = (2,2))(conv7)
    up8 = Conv2D(128, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    merge8 = Concatenate()([conv2,up8])
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    #up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    'the first adaptation'
    up9 = Conv2D(64, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    merge9 = Concatenate()([pool1, up9])
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    # 'attention part'
    # if  (attentionS==1 and  attentionC==1):
	#        o_c= attention.CAM(conv9, 64)
	#        o_s = attention.PAM(conv9, 64)
	#        conv9 = Add()([o_c, o_s])
    #
    # if  (attentionS==1 and  attentionC==0):
	#        conv9 = attention.PAM(conv9, 64)
    #
    # if  (attentionS==0 and  attentionC==1):
	#        conv9 = attention.CAM(conv9, 64)

    conv9 = Conv2D(numC, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(numC, 1, activation = 'softmax')(conv9)

    model = Model(input = inputs, output = conv10)

    return model
