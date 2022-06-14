import numpy as np

def loadCsv(loadPath):
    data = np.loadtxt(loadPath, delimiter=',', skiprows=0, encoding='utf-8-sig')
    return data

def dataset_pre(data, TL):
    Data_2 = []
    size = np.size(data, axis=0)
    for k in range(0, size):
        for j in range(k, k + TL):
            if j < size:
                Data_2.append(data[j])
            else:
                Data_2.append(data[size-1])
    Features = np.array(Data_2)
    return Features