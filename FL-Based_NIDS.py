import glob
import numpy as np
from os import path
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from tensorflow import keras
import warnings
from dataPre import loadCsv, dataset_pre

warnings.filterwarnings("ignore", category=Warning)

TL = 4

trainPath_201 = "data/UNSW_NB15_Train201.csv"
trainPath_202 = "data/UNSW_NB15_Train202.csv"
trainPath_203 = "data/UNSW_NB15_Train203.csv"
trainPath_204 = "data/UNSW_NB15_Train204.csv"
trainPath_205 = "data/UNSW_NB15_Train205.csv"

testPath_2 = 'data/UNSW_NB15_TestBin.csv'

trainData_201 = loadCsv(trainPath_201)
trainData_202 = loadCsv(trainPath_202)
trainData_203 = loadCsv(trainPath_203)
trainData_204 = loadCsv(trainPath_204)
trainData_205 = loadCsv(trainPath_205)

testData_2 = loadCsv(testPath_2)

trainData01_scaler = trainData_201[:, 0:196]
trainData02_scaler = trainData_202[:, 0:196]
trainData03_scaler = trainData_203[:, 0:196]
trainData04_scaler = trainData_204[:, 0:196]
trainData05_scaler = trainData_205[:, 0:196]

testData_scaler = testData_2[:, 0:196]

scaler = MinMaxScaler()
scaler.fit(trainData01_scaler)
trainData01_scaler = scaler.transform(trainData01_scaler)
scaler.fit(trainData02_scaler)
trainData02_scaler = scaler.transform(trainData02_scaler)
scaler.fit(trainData03_scaler)
trainData03_scaler = scaler.transform(trainData03_scaler)
scaler.fit(trainData04_scaler)
trainData04_scaler = scaler.transform(trainData04_scaler)
scaler.fit(trainData05_scaler)
trainData05_scaler = scaler.transform(trainData05_scaler)

scaler.fit(testData_scaler)
testData_scaler = scaler.transform(testData_scaler)

x_train01 = dataset_pre(trainData01_scaler, TL)
x_train01 = np.reshape(x_train01, (-1, TL, 196))
x_train02 = dataset_pre(trainData02_scaler, TL)
x_train02 = np.reshape(x_train02, (-1, TL, 196))
x_train03 = dataset_pre(trainData03_scaler, TL)
x_train03 = np.reshape(x_train03, (-1, TL, 196))
x_train04 = dataset_pre(trainData04_scaler, TL)
x_train04 = np.reshape(x_train04, (-1, TL, 196))
x_train05 = dataset_pre(trainData05_scaler, TL)
x_train05 = np.reshape(x_train05, (-1, TL, 196))

x_test = dataset_pre(testData_scaler, TL)
x_test = np.reshape(x_test, (-1, TL, 196))

# Label
y_train01 = trainData_201[:,196]
y_train02 = trainData_202[:,196]
y_train03 = trainData_203[:,196]
y_train04 = trainData_204[:,196]
y_train05 = trainData_205[:,196]
y_test = testData_2[:,196]

shape = np.size(x_train01, axis=2)

def nids_model01(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    model.fit(x_train01, y_train01, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)

    m = model.get_weights()
    np.save('Server/Server1', m)
    return model

def nids_model02(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    model.fit(x_train02, y_train02, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)

    m = model.get_weights()
    np.save('Server/Server2', m)
    return model

def nids_model03(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    model.fit(x_train03, y_train03, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)

    m = model.get_weights()
    np.save('Server/Server3', m)
    return model


def nids_model04(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    model.fit(x_train04, y_train04, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)

    m = model.get_weights()
    np.save('Server/Server4', m)
    return model

def nids_model05(shape, serverbs, serverepochs):
    if path.exists("CentralServer/fl_model.h5"):
        print("FL model exists...\nLoading model...")
        model = keras.models.load_model("CentralServer/fl_model.h5")
    else:
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    model.fit(x_train05, y_train05, batch_size=serverbs, epochs=serverepochs,
                            validation_data=(x_test, y_test), verbose=2, shuffle=True)

    m = model.get_weights()
    np.save('Server/Server5', m)

    return model

def load_models():
    arr = []
    models = glob.glob("Server/*.npy")
    for i in models:
        arr.append(np.load(i, allow_pickle=True))

    return np.array(arr)

def fl_average():

    arr = load_models()
    fl_avg = np.average(arr, axis=0)

    return fl_avg

def build_model(avg):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(TL, shape)))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    model.set_weights(avg)
    print("FL Model Ready!")

    return model

def evaluate_model(model, x_test, y_test):
    print('Test Num:', len(y_test))
    score = model.evaluate(x_test, y_test, batch_size=200000, verbose=0)
    print('Score:', score)

def save_fl_model(model):
    model.save("CentralServer/fl_model.h5")

def model_fl():
    avg = fl_average()
    model = build_model(avg)
    evaluate_model(model, x_test, y_test)
    save_fl_model(model)

fl_epochs = 300

for i in range(fl_epochs):

    model1 = nids_model01(shape, 500, 1)
    model2 = nids_model02(shape, 500, 1)
    model3 = nids_model03(shape, 500, 1)
    model4 = nids_model04(shape, 500, 1)
    model5 = nids_model05(shape, 500, 1)
    model_fl()
    print('Epoch:', i)

    K.clear_session()