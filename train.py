#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/8 13:10
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction：机器学习lab3（py2.7代码，不兼容3的地方已注释）√ ━━━━━☆*°☆*°
"""
from PIL import Image
import numpy as np
from feature import *
import os
from sklearn.cross_validation import train_test_split
from sklearn import tree
import ensemble
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import cPickle
# 不采用pickle存储和读取，速度太慢了


# 包含预处理为24*24的灰度图并提取特征，保存为.npy格式的文件方便下次读取
# 测试中pickle存储和读取速度比较慢，所以决定采用numpy的array存储
def save_images_2_array():
    face_dir = './datasets/original/face/'
    nonface_dir = './datasets/original/nonface/'
    face_features_flag = False
    nonface_features_flag = False
    cnt = 0
    for root, dirs, files in os.walk(face_dir, topdown=False):
        for file in files:
            im = Image.open(face_dir + file).convert('L')
            im = im.resize((24, 24))
            im_array = np.array(im)
            face_feature = NPDFeature(im_array).extract()
            if not face_features_flag:
                face_features = np.array(face_feature)
                face_features_flag = True
            else:
                face_features = np.row_stack((face_features, face_feature))
            cnt += 1
            print cnt
    np.save('face_feature_data_array', face_features)
    for root, dirs, files in os.walk(nonface_dir, topdown=False):
        for file in files:
            im = Image.open(nonface_dir + file).convert('L')
            im = im.resize((24, 24))
            im_array = np.array(im)
            nonface_feature = NPDFeature(im_array).extract()
            if not nonface_features_flag:
                nonface_features = np.array(nonface_feature)
                nonface_features_flag = True
            else:
                nonface_features = np.row_stack((nonface_features, nonface_feature))
            cnt += 1
            print cnt
    np.save('nonface_feature_data_array', nonface_features)

# 从npy文件中读取数据集正负样本并打好标签
def get_Data_set():
    nonface_X = np.load('nonface_feature_data_array.npy')
    face_X = np.load('face_feature_data_array.npy')
    X = face_X
    X = np.row_stack((X, nonface_X))
    y = [1 for t in range(face_X.shape[0])]
    y.extend([-1 for t in range(nonface_X.shape[0])])
    y = np.array(y)
    # print X.shape, y.shape
    return X, y

# 绘制模型预测准确率图(以基分类器的数量为x坐标，准确率为y坐标)
def draw_predict(y):
    plt.figure(figsize=(8, 6))
    plt.xlabel('The number of Base Classifier')
    plt.ylabel('Accuracy on Validation Set')
    plt.plot(range(2, 10), y, 'o-', label=u"Tree_depth = 4")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # 包含预处理为24*24的灰度图并提取特征，保存为.npy格式的文件方便下次读取，只需要运行一次
    # save_images_2_array()
    # 已经存储在nonface_feature_data_array.npy和face_feature_data_array.npy了，可以直接读取
    X, y = get_Data_set()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=214)
    DTC = tree.DecisionTreeClassifier(max_depth=2)
    # 下面为训练一个新的模型，基分类器的数量从2到10
    # 为了加深迭代过程理解，此处决策树的深度设置为2或4
    # for i in range(2, 10):
    #     My_AdaBoost = ensemble.AdaBoostClassifier(DTC, i)
    #     My_AdaBoost.fit(X_train, y_train)
    #     My_AdaBoost.predict(X_val)
    #     My_AdaBoost.is_good_enough(y_val)
    #     My_AdaBoost.save(My_AdaBoost, '2_val_' + str(i) + '_model')

    # 下面为读取已训练好的模型并进行预测
    My_AdaBoost = ensemble.AdaBoostClassifier(DTC)
    f = open('Tree_depth_4_reports.txt', 'a')
    acc = []
    for i in range(2, 10):
        My_model = My_AdaBoost.load('model/4_' + str(i) + '_model')
        My_model.predict(X_train)
        My_model.is_good_enough(y_train)
        y_predict = My_model.predict(X_val)
        accuray = accuracy_score(y_val, y_predict)
        acc.append(accuray)
        # print accuray
        report = My_model.is_good_enough(y_val)
        f.write('Classifier_Num = ' + str(i) + ':\n' + str(report) + '\n\n')
    f.close()
    draw_predict(acc)

