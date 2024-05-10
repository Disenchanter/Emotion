from get_fusion import getData
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from Confusion_matrix import plot_confusion_matrix

EMOTION_LABEL = {'NE': 0, 'IJ': 1, 'MJ': 2, 'IA': 3, 'MA': 4, 'IS': 5, 'MS': 6}

C = 19
data_feature_Acoustics, data_feature_VocalKinematics, data_feature_Fusion, data_labels = getData()
factors = ["Acoustics", "VocalKinematics", "Fusion"]
data_feature = []
for factor in factors:
    if factor == "Acoustics":
        data_feature = data_feature_Acoustics
    if factor == "VocalKinematics":
        data_feature = data_feature_VocalKinematics
    if factor == "Fusion":
        data_feature = data_feature_Fusion
    X_train, Y_train, X_test, Y_test = train_test_split(data_feature, data_labels,
                                                        test_size=0.2, shuffle=True, random_state=20010521)
    clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=C, gamma=0.0001)
    clf.fit(X_train, X_test)
    print('Train Over')
    acc = sklearn.metrics.accuracy_score(clf.predict(Y_train), Y_test)
    print(acc)
    if factor == "Acoustics":
        plot_confusion_matrix(Y_test, clf.predict(Y_train), title="SVM_声学", save_flg=True)
    if factor == "VocalKinematics":
        plot_confusion_matrix(Y_test, clf.predict(Y_train), title="SVM_发声运动学", save_flg=True)
    if factor == "Fusion":
        plot_confusion_matrix(Y_test, clf.predict(Y_train), title="SVM_数据级融合", save_flg=True)
