import os
import numpy as np
from random import random, randint

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

from mylib.RocAuc import RocAucEvaluation
from mylib.densenet import get_compiled

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 变量初始化
NUM_CLASSES = 2
batch_size = 32
epoch = 10
ckg = 32
loadmodel  = 1 # 是否加载权重
trainmodel = 0 # 是否进行训练

if trainmodel == 1:
    # 加载基础权重，训练，预测
    model_path = 'weights/weights.base.h5'
else:
    # 加载权重，仅预测
    model_path = 'weights/weights.70762.h5'

# 数据读取
h_ckg = int(ckg / 2)

x_train = np.ones((465, ckg, ckg, ckg))
x_test = np.ones((117, ckg, ckg, ckg))

# 读取训练集
i = 0
path = "dataset/train_val"  # 待读取的文件夹
path_list = os.listdir(path)
path_list.sort()  # 对读取的路径进行排序
for filename in path_list:
    tmp = np.load(os.path.join(path, filename))

    voxel = tmp['voxel']
    seg = tmp['seg']

    x_train[i] = (voxel * seg)[50 - h_ckg:50 + h_ckg, 50 - h_ckg:50 + h_ckg, 50 - h_ckg:50 + h_ckg]

    i = i + 1
# 读取测试集
i = 0
path = "dataset/test"  # 待读取的文件夹
path_list = os.listdir(path)
path_list.sort()  # 对读取的路径进行排序
for filename in path_list:
    tmp = np.load(os.path.join(path, filename))

    voxel = tmp['voxel']
    seg = tmp['seg']

    x_test[i] = (voxel * seg)[50 - h_ckg:50 + h_ckg, 50 - h_ckg:50 + h_ckg, 50 - h_ckg:50 + h_ckg]

    i = i + 1
# 读取训练集label
path = 'dataset/train_val.csv'
y_train = np.loadtxt(path, int, delimiter=",", skiprows=1, usecols=1)

# 验证集
x_val = x_train.copy()
y_val = y_train.copy()

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape)

# 数据增强
# ------------------------------------------------------------------------------------------------------------------------------------------
'''
tmp_x = x_train
tmp_y = y_train

x_train = np.ones((465*3, ckg, ckg, ckg))
y_train = np.ones(465*3)

for i in range (0,465):

    tmp_array1 = tmp_x[i]
    for j in range(0,h_ckg):
        tmp_array1[j, :, :],tmp_array1[ckg-1-j, :, :] = tmp_array1[ckg-1-j, :, :],tmp_array1[j, :, :]

    tmp_array2 = tmp_array1
    for j in range(0,h_ckg):
        tmp_array2[:, j, :],tmp_array2[:, ckg-1-j, :] = tmp_array2[:, ckg-1-j, :],tmp_array2[:, j, :]

    tmp_array3 = tmp_array1
    for j in range(0,h_ckg):
        tmp_array3[:, :, j],tmp_array3[:, :, ckg-1-j] = tmp_array3[:, :, ckg-1-j],tmp_array3[:, :, j]

    x_train[i+465*0] = tmp_array1
    x_train[i+465*1] = tmp_array2
    x_train[i+465*2] = tmp_array3

y_train[465*0:465*1] = tmp_y[0:465]
y_train[465*1:465*2] = tmp_y[0:465]
y_train[465*2:465*3] = tmp_y[0:465]
'''
# ------------------------------------------------------------------------------------------------------------------------------------------

tmp_x = x_train.copy()
tmp_y = y_train.copy()

x_train = np.ones((465 * 3, ckg, ckg, ckg))
y_train = np.ones(465 * 3)

for i in range(0, 465):
    tmp_array1 = tmp_x[i].copy()
    for j in range(0, h_ckg):
        tmp_array1[j, :, :], tmp_array1[ckg - 1 - j, :, :] = tmp_array1[ckg - 1 - j, :, :], tmp_array1[j, :, :]
    for j in range(0, h_ckg):
        tmp_array1[:, j, :], tmp_array1[:, ckg - 1 - j, :] = tmp_array1[:, ckg - 1 - j, :], tmp_array1[:, j, :]

    tmp_array2 = tmp_x[i].copy()
    for j in range(0, h_ckg):
        tmp_array2[:, j, :], tmp_array2[:, ckg - 1 - j, :] = tmp_array2[:, ckg - 1 - j, :], tmp_array2[:, j, :]
    for j in range(0, h_ckg):
        tmp_array2[:, :, j], tmp_array2[:, :, ckg - 1 - j] = tmp_array2[:, :, ckg - 1 - j], tmp_array2[:, :, j]

    tmp_array3 = tmp_x[i].copy()
    for j in range(0, h_ckg):
        tmp_array3[:, :, j], tmp_array3[:, :, ckg - 1 - j] = tmp_array3[:, :, ckg - 1 - j], tmp_array3[:, :, j]
    for j in range(0, h_ckg):
        tmp_array3[j, :, :], tmp_array3[ckg - 1 - j, :, :] = tmp_array3[ckg - 1 - j, :, :], tmp_array3[j, :, :]

    x_train[i + 465 * 0] = tmp_array1.copy()
    x_train[i + 465 * 1] = tmp_array2.copy()
    x_train[i + 465 * 2] = tmp_array3.copy()

y_train[465 * 0:465 * 1] = tmp_y.copy()
y_train[465 * 1:465 * 2] = tmp_y.copy()
y_train[465 * 2:465 * 3] = tmp_y.copy()

# ------------------------------------------------------------------------------------------------------------------------------------------
'''
# mixup
tmp_x = x_train.copy()
tmp_y = y_train.copy()

x_train = np.ones((800, ckg, ckg, ckg))
y_train = np.ones(800)

t = 0.5

for i in range (0,800):
    m = randint(0,465-1)
    n = randint(0,465-1)
    #t = round(random(),2)

    x_train[i] = (t*tmp_x[m] + (1-t)*tmp_x[n]).copy()
    y_train[i] = (t*tmp_y[m] + (1-t)*tmp_y[n]).copy()
'''
# ------------------------------------------------------------------------------------------------------------------------------------------

# 将数据维度进行处理
x_train = x_train.reshape(x_train.shape[0], ckg, ckg, ckg, 1)
x_val = x_val.reshape(x_val.shape[0], ckg, ckg, ckg, 1)
x_test = x_test.reshape(x_test.shape[0], ckg, ckg, ckg, 1)

y_train = to_categorical(y_train, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape)

# compile model
model = get_compiled()

if loadmodel == 1:
    model.load_weights(model_path)

# train
if trainmodel == 1:
    RocAuc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
    checkpointer = ModelCheckpoint(filepath='tmp/weights.{epoch:02d}.h5', verbose=1,
                                   period=1, save_weights_only=True)

    model.fit(x_train,
              y_train,
              epochs=epoch,
              validation_data=(x_val, y_val),
              shuffle=False,
              batch_size=batch_size,
              callbacks=[RocAuc, checkpointer])

# predict
test_pre = model.predict(x_test, batch_size, verbose=1)
train_pre = model.predict(x_val, batch_size, verbose=1)

# print
print('Training accuracy: %.5f%%' % (100 * score))

# 保存预测文件，分隔符为,
col0 = np.loadtxt("sampleSubmission.csv", str, delimiter=",", skiprows=1, usecols=0)
path = "Submission.csv"
np.savetxt(path, np.column_stack((col0,test_pre[:, 1])), delimiter=',', fmt='%s', header='Id,Predicted', comments='')
print('File saved')
