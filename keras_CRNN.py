from get_CRNN import getData
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, TimeDistributed
from sklearn.model_selection import train_test_split
from Confusion_matrix import plot_confusion_matrix

EMOTION_LABEL = {'NE': 0, 'IJ': 1, 'MJ': 2, 'IA': 3, 'MA': 4, 'IS': 5, 'MS': 6}

data_feature_Acoustics, data_feature_VocalKinematics, data_labels = getData()
data_feature_VocalKinematics = data_feature_VocalKinematics / 64.0

batch_size = 32
num_classes = 7
keras.backend.set_floatx('float64')

factors = ["Acoustics", "VocalKinematics"]
data_feature = []

for factor in factors:
    model = Sequential()
    if factor == "Acoustics":
        data_feature = data_feature_Acoustics
        epochs = 20
    if factor == "VocalKinematics":
        data_feature = data_feature_VocalKinematics
        epochs = 50

    X_train, Y_train, X_test, Y_test = train_test_split(data_feature, data_labels,
                                                        test_size=0.2, shuffle=False, random_state=20010521)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='tanh',
                     input_shape=(data_feature.shape[1], data_feature.shape[2], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(512, activation='tanh'))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, activation='tanh'))
    model.add(Dense(num_classes))
    batch_size = 32

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam', metrics=['sparse_categorical_accuracy'])
    model.fit(data_feature, data_labels, epochs=epochs, batch_size=batch_size)

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

    plot_confusion_matrix(Y_test, predict, title="卷积循环神经网络_" + title, save_flg=True)


print(np.array(p1).shape)
print(np.array(p2).shape)
model = Sequential([
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(7)
])
p = np.concatenate((p1, p2))
print(np.array(p).shape)


"""
p = []
predict = []
p = p1 + p2 * 2
p = np.array(p)
predict.append(np.argmax(p, axis=1))
predict = np.array(predict)
predict = predict.flatten()
correct_prediction = np.equal(predict, Y_test)
print(np.mean(correct_prediction))
plot_confusion_matrix(Y_test, predict, title="卷积循环神经网络_决策级融合", save_flg=True)
"""