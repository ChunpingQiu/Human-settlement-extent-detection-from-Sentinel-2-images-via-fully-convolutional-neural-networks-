# @Date:   2019-12-21T17:33:42+01:00
# @Last modified time: 2020-01-27T15:00:02+01:00

# import sys
from keras import backend as K
from tensorflow.python.ops import math_ops

'define networks for training or predicitions'
def modelSelection(nn, nadam, patch_shape=(128,128,10), numClasses=2):

	dataFlow = 0
	###################################################################
	'simple net'
	if nn==12:
		import sen2IS_net as DNN
		'without BN'
		model = DNN.sen2IS_net_bn(input_size = patch_shape, numC=numClasses, ifBN=0)#DNN.sen2IS_net(input_size = patch_shape, numC=2)
		model.compile(optimizer=nadam, loss= 'binary_crossentropy', metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])
		model.summary()

   ##################################################
	'test effct of network shape'
	'width'
	if nn==26:#
		import sen2IS_net as DNN
		'without BN'
		model = DNN.sen2IS_net_wide(input_size = (128,128,10), numC=2, ifBN=0)
		model.compile(optimizer=nadam, loss= 'binary_crossentropy', metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])

	'deep newwork is not esay to train without bn?'
	'depth'
	if nn==27:
		import sen2IS_net as DNN
		'without BN'
		model = DNN.sen2IS_net_deep(input_size = (128,128,10), numC=2, depth=13)
		model.compile(optimizer=nadam, loss= 'binary_crossentropy', metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])

	if nn==28:
		import sen2IS_net as DNN
		'without BN'
		model = DNN.sen2IS_net_deep(input_size = (128,128,10), numC=2, depth=17)
		model.compile(optimizer=nadam, loss= 'binary_crossentropy', metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])
	if nn==29:
		import sen2IS_net as DNN
		'without BN'
		model = DNN.sen2IS_net_deep(input_size = (128,128,10), numC=2, depth=21)
		model.compile(optimizer=nadam, loss= 'binary_crossentropy', metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])

	##################################################

	'to test effect of weight'
	if nn==13:#
		import sen2IS_net as DNN
		model = DNN.sen2IS_net_bn(input_size = patch_shape, numC=numClasses, ifBN=0)
		model.compile(optimizer=nadam, loss= binary_crossentropy_weight, metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])

	'to test effect of BN'
	if nn==14:#
		import sen2IS_net as DNN
		model = DNN.sen2IS_net_bn(input_size = patch_shape, numC=numClasses)
		model.compile(optimizer=nadam, loss= 'binary_crossentropy', metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])

	######################################################
	'baselines, but with equavelent para?'
	if nn==31:#
		import sen2IS_net as DNN
		model = DNN.unet(input_size = patch_shape, numC=numClasses)
		model.compile(optimizer=nadam, loss= 'binary_crossentropy', metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])
		#model.summary()
	if nn==32:#
		import sys
		sys.path.insert(0, '/home/qiu/CodeSummary/0urbanMapper/HSE4RSE/image-segmentation-keras-master/image-segmentation-keras-master/keras_segmentation/models')
		from pspnet import resnet50_pspnet
		model = resnet50_pspnet(2,  input_height=128, input_width=128)
		model.compile(optimizer=nadam, loss= 'binary_crossentropy', metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])
	if nn==33:#
		import sys
		sys.path.insert(0, '/home/qiu/CodeSummary/0urbanMapper/HSE4RSE/image-segmentation-keras-master/image-segmentation-keras-master/keras_segmentation/models')
		from fcn import fcn_8_resnet50
		model = fcn_8_resnet50(2,  input_height=128, input_width=128)
		model.compile(optimizer=nadam, loss= 'binary_crossentropy', metrics=['accuracy', dice_coef, f1_m, precision_m, recall_m])
	###################################################################
	'dual attention'
	'the first submission'
	if nn==0:#
		import sys
		sys.path.insert(0, '/home/qiu/CodeSummary/MTMS4urban/model')
		import nn as DNN
		model = DNN.build_model_resFCN(patch_shape, nb_classes = (2,17), mt=0, res=0, atnS=0, atnC=0)
	# if nn==1:#
	# 	model = DNN.build_model_resFCN(patch_shape, nb_classes = (2,17), mt=0, res=0, atnS=0, atnC=1)
	# if nn==2:#
	# 	model = DNN.build_model_resFCN(patch_shape, nb_classes = (2,17), mt=0, res=0, atnS=1, atnC=0)
	if nn==3:#
		import sys
		sys.path.insert(0, '/home/qiu/CodeSummary/MTMS4urban/model')
		import nn as DNN
		model = DNN.build_model_resFCN(patch_shape, nb_classes = (2,17), mt=0, res=0, atnS=1, atnC=1)
		'first trained version'
		#modelName =  '/data/qiu/data4RSEpaper/consistentResult/4Seasons/small/' + str(2e-5) + '_' + str(3)+"_8weights.best.hdf5"


	return model, dataFlow

################################################################################
def recall_m(y_true_oneHot, y_pred_oneHot):
        y_true=K.cast(K.argmax(y_true_oneHot, axis=-1), K.floatx())
        y_pred=K.cast(K.argmax(y_pred_oneHot, axis=-1), K.floatx())

        'because class 0 is the target class'
        y_true = K.cast(math_ops.subtract(1.0, y_true), K.floatx())#
        y_pred = K.cast(math_ops.subtract(1.0, y_pred), K.floatx())


        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true_oneHot, y_pred_oneHot):

        y_true=K.cast(K.argmax(y_true_oneHot, axis=-1), K.floatx())
        y_pred=K.cast(K.argmax(y_pred_oneHot, axis=-1), K.floatx())

        'because class 0 is the target class'
        y_true = K.cast(math_ops.subtract(1.0, y_true), K.floatx())#
        y_pred = K.cast(math_ops.subtract(1.0, y_pred), K.floatx())

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def dice_coef(y_true_oneHot, y_pred_oneHot):

    y_true=K.cast(K.argmax(y_true_oneHot, axis=-1), K.floatx())
    y_pred=K.cast(K.argmax(y_pred_oneHot, axis=-1), K.floatx())

    'because class 0 is the target class'
    y_true = K.cast(math_ops.subtract(1.0, y_true), K.floatx())#
    y_pred = K.cast(math_ops.subtract(1.0, y_pred), K.floatx())

    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    # print(y_true_f.shape, y_pred_f.shape)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)#, _val_f1, _val_recall, _val_precision
#
