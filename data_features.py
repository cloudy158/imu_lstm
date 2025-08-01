import numpy as np


def featureRMS(data):
    """
    计算均方根（Root Mean Square, RMS）特征。

    参数:
    data (numpy.ndarray): 输入数据，形状为 (n_samples, n_channels)

    返回:
    numpy.ndarray: 每个通道的 RMS 特征值，形状为 (n_channels,)
    """
    return np.sqrt(np.mean(data ** 2, axis=0))


def featureMAV(data):
    """
    计算平均绝对值（Mean Absolute Value, MAV）特征。

    参数:
    data (numpy.ndarray): 输入数据，形状为 (n_samples, n_channels)

    返回:
    numpy.ndarray: 每个通道的 MAV 特征值，形状为 (n_channels,)
    """
    return np.mean(np.abs(data), axis=0)


def featureWL(data):
    """
    计算波形长度（Waveform Length, WL）特征。

    参数:
    data (numpy.ndarray): 输入数据，形状为 (n_samples, n_channels)

    返回:
    numpy.ndarray: 每个通道的 WL 特征值，形状为 (n_channels,)
    """
    return np.sum(np.abs(np.diff(data, axis=0)), axis=0) / data.shape[0]


def featureZC(data, threshold=10e-7):
    """
    计算零交叉率（Zero Crossing Rate, ZC）特征。

    参数:
    data (numpy.ndarray): 输入数据，形状为 (n_samples, n_channels)
    threshold (float): 用于确定零交叉的阈值，默认为 10e-7

    返回:
    numpy.ndarray: 每个通道的 ZC 特征值，形状为 (n_channels,)
    """
    numOfZC = []
    channel = data.shape[1]
    length = data.shape[0]

    for i in range(channel):
        count = 0
        for j in range(1, length):
            diff = data[j, i] - data[j - 1, i]
            mult = data[j, i] * data[j - 1, i]

            if np.abs(diff) > threshold and mult < 0:
                count = count + 1
        numOfZC.append(count / length)
    return np.array(numOfZC)


def featureSSC(data, threshold=10e-7):
    """
    计算符号对称系数（Symmetric Slope Change, SSC）特征。

    参数:
    data (numpy.ndarray): 输入数据，形状为 (n_samples, n_channels)
    threshold (float): 用于确定符号变化的阈值，默认为 10e-7

    返回:
    numpy.ndarray: 每个通道的 SSC 特征值，形状为 (n_channels,)
    """
    numOfSSC = []
    channel = data.shape[1]
    length = data.shape[0]

    for i in range(channel):
        count = 0
        for j in range(2, length):
            diff1 = data[j, i] - data[j - 1, i]
            diff2 = data[j - 1, i] - data[j - 2, i]
            sign = diff1 * diff2

            if sign > 0:
                if (np.abs(diff1) > threshold or np.abs(diff2) > threshold):
                    count = count + 1
        numOfSSC.append(count / length)
    return np.array(numOfSSC)