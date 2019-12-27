import os
import numpy as np
from mylib.densenet import get_compiled
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 变量初始化
batch_size = 32
ckg = 32
h_ckg = int(ckg / 2)

x_test = np.ones((117, ckg, ckg, ckg))
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

x_test = x_test.reshape(x_test.shape[0], ckg, ckg, ckg, 1)

# compile model
model = get_compiled()

model_path1 = 'weights/weights.70557.h5'
model_path2 = 'weights/weights.70674.h5'
model_path3 = 'weights/weights.70762.h5'

# predict
model.load_weights(model_path1)
test1 = model.predict(x_test, batch_size, verbose=1)

model.load_weights(model_path2)
test2 = model.predict(x_test, batch_size, verbose=1)

model.load_weights(model_path3)
test3 = model.predict(x_test, batch_size, verbose=1)

# (70557)+(70674)=0.72111
# (70557)+(70674)+(70762)=0.73313

test = (test1 + test2 + test3) / 3

# 保存预测文件，分隔符为,
col0 = np.loadtxt("sampleSubmission.csv", str, delimiter=",", skiprows=1, usecols=0)
path = "Submission.csv"
np.savetxt(path, np.column_stack((col0,test[:, 1])), delimiter=',', fmt='%s', header='Id,Predicted', comments='')
print('File saved')
