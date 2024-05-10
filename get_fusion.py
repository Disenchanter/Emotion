import librosa
import random
import scipy.io as io
import os
import numpy as np

k = 9
path = r'C:\Users\LZ\Desktop\Coding\tensorflow1\STEM-E2Va'
path_ = r'G:\LiZhi\tensorflow1\t'
EMOTION_LABEL = {'NE': 0, 'IJ': 1, 'MJ': 2, 'IA': 3, 'MA': 4, 'IS': 5, 'MS': 6}


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dict = {}
    for key in dict_key_ls:
        new_dict[key] = dicts.get(key)
    return new_dict


def getData(mfcc_feature_num=20):
    data_feature_Acoustics = []
    data_feature_VocalKinematics = []
    data_feature_Fusion = []
    data_labels = []
    file_path = {'': ''}
    person_dirs = os.listdir(path)
    for person in person_dirs:
        if person.endswith('README'):
            continue
        wavfile_path = os.path.join(path, person, 'wavfiles')
        wavfile = os.listdir(wavfile_path)

        matfile_path = os.path.join(path, person, 'matfiles')
        matfile = os.listdir(matfile_path)

        for file in wavfile:
            if not file.endswith('.wav'):
                continue
            wav = os.path.join(path, person, 'wavfiles', file)
            file_path[wav] = file.strip('.wav')

    file_path = random_dic(file_path)

    for wav_file in file_path:
        if not wav_file.endswith('.wav'):
            continue
        # 声学
        y, sr = librosa.load(wav_file)
        mfcc_feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_feature_num)
        zcr_feature = librosa.feature.zero_crossing_rate(y=y)
        energy_feature = librosa.feature.rms(y=y)
        mfccadd = []
        time = len(mfcc_feature)
        for i in range(k):
            mfccadd.extend(mfcc_feature[:, int(time * i / k)])
        mfccadd.extend(mfcc_feature[:, time - 1])

        zcr_feature = zcr_feature.flatten()
        energy_feature = energy_feature.flatten()
        zcr_feature = np.array([np.mean(zcr_feature)])
        energy_feature = np.array([np.mean(energy_feature)])
        # 发声运动学
        mat = io.loadmat(wav_file.replace('wav', 'mat'))
        key = file_path[wav_file].strip(".wav")
        mat = mat[key]
        time = len(mat)
        matadd = []
        for i in range(k):
            matadd.extend(mat[int(time * i / k)])
        matadd.extend(mat[time - 1])

        data_feature_Acoustics.append(np.concatenate((mfccadd, zcr_feature, energy_feature)))
        data_feature_Fusion.append(np.concatenate((mfccadd, zcr_feature, energy_feature, matadd)))
        data_feature_VocalKinematics.append(matadd)
        # 情感类别
        for key in EMOTION_LABEL:
            if key in wav_file:
                data_labels.append(EMOTION_LABEL[key])

    data_feature_Acoustics = np.array(data_feature_Acoustics)
    data_feature_VocalKinematics = np.array(data_feature_VocalKinematics)
    data_feature_Fusion = np.array(data_feature_Fusion)
    data_labels = np.array(data_labels)
    return data_feature_Acoustics, data_feature_VocalKinematics, data_feature_Fusion, data_labels

# getData()
