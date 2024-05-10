from __future__ import print_function
import itertools
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title="Undefined", cmap=plt.cm.Blues, save_flg=False):
    matplotlib.rc("font", family='Microsoft YaHei')
    classes = ['NE', 'IJ', 'MJ', 'IA', 'MA', 'IS', 'MS']
    labels = [0, 1, 2, 3, 4, 5, 6]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=10)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('真实值', fontsize=8)
    plt.xlabel('预测值', fontsize=8)
    if save_flg:
        plt.savefig("./"+title+"的混淆矩阵.png")
    print(title+"_ConfusionMatrix has been completed.")
    # plt.show()
