from get_fusion import getData
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from Confusion_matrix import plot_confusion_matrix

EMOTION_LABEL = {'NE': 0, 'IJ': 1, 'MJ': 2, 'IA': 3, 'MA': 4, 'IS': 5, 'MS': 6}

data_feature_Acoustics, data_feature_VocalKinematics, data_feature_Fusion, data_labels = getData()
data_feature_VocalKinematics = data_feature_VocalKinematics / 64.0
factors = ["Acoustics", "VocalKinematics", "Fusion"]
data_feature = []
for factor in factors:
    if factor == "Acoustics":
        data_feature = data_feature_Acoustics
        model = Sequential([
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(7)
        ])
    if factor == "VocalKinematics":
        data_feature = data_feature_VocalKinematics
        model = Sequential([
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(7)
        ])
    if factor == "Fusion":
        data_feature = data_feature_Fusion
        model = Sequential([
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(7)
        ])

    X_train, Y_train, X_test, Y_test = train_test_split(data_feature, data_labels,
                                                        test_size=0.2, shuffle=True, random_state=20010521)

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    model.fit(X_train, X_test, epochs=100)
    print("Start test")
    model.evaluate(Y_train, Y_test)

    Y_predict = model.predict(Y_train)

    predict = []
    p = []

    for i in range(np.array(Y_predict).shape[0]):
        for j in range(np.array(Y_predict).shape[1]):
            if np.sum(Y_predict, axis=1)[i] > 0:
                p.append(Y_predict[i][j] / (np.sum(Y_predict, axis=1))[i])
            if np.sum(Y_predict, axis=1)[i] < 0:
                p.append(Y_predict[i][j] / (np.sum(Y_predict, axis=1))[i] * -1)

    predict.append(np.argmax(Y_predict, axis=1))
    predict = np.array(predict)
    predict = predict.flatten()

    p = np.array(p)
    p = p.reshape(np.array(Y_predict).shape[0], np.array(Y_predict).shape[1])
    if factor == "Acoustics":
        p1 = p
        title = "声学"
    if factor == "VocalKinematics":
        p2 = p
        title = "发声运动学"
    if factor == "Fusion":
        title = "数据级融合"

    plot_confusion_matrix(Y_test, predict, title="前馈神经网络_"+title, save_flg=True)



p = []
predict = []
p = p1 + p2 * 2
p = np.array(p)
predict.append(np.argmax(p, axis=1))
predict = np.array(predict)
predict = predict.flatten()
correct_prediction = np.equal(predict, Y_test)
print(np.mean(correct_prediction))
plot_confusion_matrix(Y_test, predict, title="前馈神经网络_决策级融合", save_flg=True)
