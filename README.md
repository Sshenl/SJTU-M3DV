# -M3DV-
This is the course assignment of EE369 machine learning 2019 autumn course of Shanghai Jiaotong University, 3D medical image classification (M3DV) Kaggle inclass competition.
This is a classification project of pulmonary nodules. We use a deep learning system to complete this project, and the final score on Kaggle is 0.73313.
# Code Structure
* [`mylib/`](mylib/):
    * [`densenet.py`](mylib/densenet.py): 3D *DenseNet* models
    * [`RocAuc`](mylib/RocAuc.py): Roc-Auc evaluation.
* [`main.py`](main.py): Compile model, train and predict.
* [`test.py`](test.py): Get "Submission.csv" file, corresponding to my best score in Kaggle.
