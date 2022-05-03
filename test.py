import random
import keras
import keras.callbacks
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time

from keras import backend as K
from PIL import Image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Softmax, Activation, Dropout
from tensorflow.keras.layers import Conv2D,MaxPool2D,add,UpSampling2D,Multiply
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import os
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tensorflow.keras.applications.resnet import ResNet50
from keras import regularizers

#from keras.utils.training_utils import multi_gpu_model
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr



model_file = 'Result_saving/entire_model.json'
best_weight_file = './Result_saving/Mymodel_12.h5'
test_data_file = './Dataset_process/test.csv'
path_to_images = './images' #'/home/yann/桌面/AVA_images/AVA_dataset/images/images'
path_to_test_result_saving = './Test_Performance'

width = 224
height = 224

def LCC(x, y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    return pearsonr(x, y)[0]

def SRCC(x, y):
    return spearmanr(x, y)[0]

def list_from_file(_file):
    csv_file = _file
    anno = pd.read_csv(csv_file, sep=' ', header=None)

    li = []
    for idx, row in anno.iterrows():
        l = []
        img_id = int(row[0])
        l.append(img_id)

        score_distribution = row[1:11]
        #score_distribution = score_distribution.to_numpy()
        score_distribution = np.array(score_distribution)
     
        score_distribution = score_distribution / sum(score_distribution)
        score_distribution = score_distribution * 10000
        score_distribution = score_distribution.astype(int)
        score_distribution = score_distribution.astype(float)
        score_distribution = score_distribution / 10000
        for i in range(10):
            l.append(score_distribution[i])

        m = row[11:13]

        #m = m.to_numpy()
        m = np.array(m)

        l.append(int(m[0]))
        l.append(int(m[1]))
        li.append(l)
    return li

#with open(model_file, 'r') as json_file:
#    loaded_model_json = json_file.read()
#loaded_model = model_from_json(loaded_model_json)

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

def Conv_block(input_x, block_id, inplanes, planes, flag=0, dim=None):

    name = 'Block_' + block_id
    residual = input_x
    stride = flag + 1

    x = Conv2D(inplanes, kernel_size=1, use_bias=False, name=name + '_' + 'Conv1', kernel_initializer='random_normal')(input_x)
    x = BatchNormalization(name=name + '_' + 'conv1_BatchNorm')(x)
    x = Activation('relu')(x)

    x = Conv2D(inplanes, kernel_size=3, use_bias=False, padding='SAME', strides=stride, name=name + '_' + 'Conv2',kernel_initializer='random_normal')(x)
    x = BatchNormalization(name=name + '_' + 'conv2_BatchNorm')(x)
    x = Activation('relu')(x)

    x = Conv2D(planes, kernel_size=1, use_bias=False, name=name + '_' + 'Conv3', kernel_initializer='random_normal')(x)
    x = BatchNormalization(name=name + '_' + 'conv3_BatchNorm')(x)

    if flag != 0 or dim != None:
        residual = Conv2D(dim * 2, kernel_size=1, use_bias=False, strides=stride, name=name + '_' + 'Conv_residual', kernel_initializer='random_normal')(input_x)
        residual = BatchNormalization(name=name + '_' + 'Conv_residual_BatchNorm')(residual)

    x = add([residual, x])
    y = Activation('relu')(x)
    return y


base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
base_model.load_weights('basemodel_resnet50.h5')
# base_model = load_model('./base_model_resnext.h5')

for layer in base_model.layers:
    layer.trainable = False

x2 = base_model.layers[38].output  # 56x56x256
x3 = base_model.layers[80].output  # 28x28x512
x4 = base_model.layers[142].output  # 14x14x1024
x5 = base_model.layers[174].output  # 7x7x2048

#注意力加权
x5_temp = UpSampling2D(size=(2, 2))(x5)
x5_temp = Conv_block(x5_temp, '3', 512, 1024, dim=512)
w4 = keras.activations.sigmoid(x5_temp)
x4_plus = Multiply()([w4, x4])
x4_plus = add([x4, x4_plus])

x4_temp = UpSampling2D(size=(2, 2))(x4_plus)
x4_temp = Conv_block(x4_temp, '2', 256, 512, dim=256)
w3 = keras.activations.sigmoid(x4_temp)
x3_plus = Multiply()([w3, x3])
x3_plus = add([x3, x3_plus])

x3_temp = UpSampling2D(size=(2, 2))(x3_plus)
x3_temp = Conv_block(x3_temp, '1', 128, 256, dim=128)
w2 = keras.activations.sigmoid(x3_temp)
x2_plus = Multiply()([w2, x2])
x2_plus = add([x2, x2_plus])

#特征融合

x2_plus = Conv_block(x2_plus, '4', 128, 512, dim=256)
x3_tempp = UpSampling2D(size=(2, 2))(x3_plus)
step1 = add([x2_plus, x3_tempp])
y1 = Conv_block(step1, '5', 256, 1024, flag=1, dim=512)

x4_tempp = UpSampling2D(size=(2, 2))(x4_plus)
step2 = add([y1, x4_tempp])
y2 = Conv_block(step2, '6', 512, 2048, flag=1, dim=1024)

x5_tempp = UpSampling2D(size=(2, 2))(x5)
step3 = add([y2, x5_tempp])
y3 = Conv_block(step3, '7', 512, 2048, flag=1, dim=1024)


# 全连接，多任务
x = GlobalAveragePooling2D()(y3)
x1 = Dense(1024, bias_initializer=keras.initializers.Ones(), kernel_initializer='random_normal')(x)
x1=BatchNormalization(name="FC/BatchNorm1")(x1)
x1 = Activation('relu')(x1)
# x1 = Dense(512, bias_initializer=keras.initializers.Ones())(x)
# x1=BatchNormalization(name="FC/BatchNorm2")(x1)#add
# x1 = Activation('relu')(x1)
x = Dropout(0.25)(x1)
out1 = Dense(10, activation='softmax', name="out1", bias_initializer=keras.initializers.Ones(), kernel_initializer='random_normal')(x)

model = Model(inputs=base_model.input, outputs=out1)

#model.load_weights(finetune_best_weight_file)

def get_lr_metric(optimizer):  # printing the value of the learning rate
    def func_lr(y_true, y_pred):
        return optimizer.lr

    return func_lr

optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.001/80)
lr_metric = get_lr_metric(optimizer)

model.compile(optimizer=optimizer,
              loss={'out1': earth_mover_loss},
              metrics=["accuracy", lr_metric])

# for i in range(len(loaded_model.layers)):
#     if loaded_model.layers[i].name == 'out1':
#         print(i)
#         print(loaded_model.layers[i].name)
#         print(loaded_model.layers)
#         print(loaded_model.layers[i].output.shape)
# print(loaded_model.layers[354].output.shape)

model.load_weights(best_weight_file)

print('Params has been loaded.')

#for i in range(len(loaded_model.layers)):
#    print(i)
#    print(loaded_model.layers[i])
#    print(loaded_model.layers[i].output)
#input()

model = Model(inputs=model.input, outputs=model.output)

test_list = list_from_file(test_data_file)

predicted_mean_list = []
predicted_std_dev_list = []
groundtruth_mean_list = []
groundtruth_std_dev_list = []

for i in range(0, len(test_list)):
    # if i%100==0:
    #     print('{} images have done.'.format(i))


    im = Image.open(path_to_images + '/' + str(test_list[i][0]) + '.jpg')
    im = im.resize((width, height), Image.BILINEAR)
    im = np.array(im).astype(float)
    im = im / 255
    if im.shape == (224, 224):
        im = np.stack((im, im, im), axis=2)
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

    # im = tf.cast(im, tf.float32)
    im = tf.expand_dims(im, axis=0)

    scores = model.predict(im, batch_size=1, verbose=0)
    print(scores)
    #print(type(scores))
    #print(np.array(scores).shape)

    # 计算真实分布的均值和标准差
    gt_l = test_list[i][1:11]
    mean_gt, std_gt = 0.0, 0.0
    for m, item in enumerate(gt_l, 1):
        mean_gt += m * item
    for n, item in enumerate(gt_l, 1):
        std_gt += item * (n - mean_gt) ** 2
    std_gt = std_gt ** 0.5
    groundtruth_mean_list.append(mean_gt)
    groundtruth_std_dev_list.append(std_gt)

    # 筛选数据进行测试
    # if(mean_gt <= 5.8 and mean_gt >=4.2):
    #     continue

    #计算预测分布的均值和标准差
    pre_distribution = scores[0]
    #print(scores[0][0])
    mean, std = 0.0, 0.0
    for m, item in enumerate(pre_distribution, 1):
        mean += m * item
    for n, item in enumerate(pre_distribution, 1):
        std += item * (n - mean) ** 2
    std = std ** 0.5
    predicted_mean_list.append(mean)
    predicted_std_dev_list.append(std)

    if not os.path.exists(path_to_test_result_saving):
        os.makedirs(path_to_test_result_saving)
    
    #print(test_list[i][0])
    
    temp_imgid = test_list[i][0]
    temp_imgid = str(temp_imgid)
    #print(mean,std,mean_gt)
    temp_result = ' mean: %.3f | std: %.3f | GT: %.3f\n' % (mean, std, mean_gt)
    #print(temp_result)
    print(temp_imgid + ' mean: %.3f | std: %.3f | GT: %.3f\n' % (mean, std, mean_gt))

    #with open(os.path.join(path_to_test_result_saving, 'pred.txt'), 'a') as f:
    #    f.write(temp_imgid + ' mean: %.3f | std: %.3f | GT: %.3f\n' % (mean, std, mean_gt))




    # print(scores[0]) # score_list
    # print(type(scores)) # list type.
    # print('===================================')

hit = 0
TP, FN, FP, TN = 0, 0, 0, 0
for i in  range(len(predicted_mean_list)):
    if (predicted_mean_list[i] > 5 and groundtruth_mean_list[i] > 5):
        TP += 1
    elif (predicted_mean_list[i] <= 5 and groundtruth_mean_list[i] > 5):
        FN += 1
    elif (predicted_mean_list[i] > 5 and groundtruth_mean_list[i] <= 5):
        FP += 1
    else:
        TN += 1

    hit = TP + TN
mse = mean_squared_error(predicted_mean_list, groundtruth_mean_list)
mean_srcc = SRCC(predicted_mean_list, groundtruth_mean_list)
mean_lcc = LCC(predicted_mean_list, groundtruth_mean_list)
std_dev_srcc = SRCC(predicted_std_dev_list, groundtruth_std_dev_list)
std_dev_lcc = LCC(predicted_std_dev_list, groundtruth_std_dev_list)

print('TP = {}'.format(TP))
print('FN = {}'.format(FN))
print('FP = {}'.format(FP))
print('TN = {}'.format(TN))

print('===============================================================')
print('Binary Classification Performance : ')
print('Accuracy of Binary Classification : {}'.format(hit / len(predicted_mean_list)))
print('Precision of Binary Classification : {}'.format(TP / (TP + FP)))
print('Recall of Binary Classication : {}'.format(TP / (TP + FN)))
# 由于预测的对象是美学得分，不是二分类问题中的概率，所以无法做出ROC曲线。
print('===============================================================')
print('Regression Performance:')
print('MSE : {}'.format(mse))
print('SRCC_mean : {}'.format(mean_srcc))
print('PLCC_mean : {}'.format(mean_lcc))
print('SRCC_std.dev : {}'.format(std_dev_srcc))
print('PLCC_std.dev : {}'.format(std_dev_lcc))


