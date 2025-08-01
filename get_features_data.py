import numpy as np
import pandas as pd
from data_features import  *

# 假设 df 是你的原始数据 DataFrame
df = pd.read_csv('3.csv', header=0)

df = df.values

# 初始化特征数据和标签列表
featureData = []  # 用于存储所有特征数据
labels = []  # 用于存储所有标签数据
timeWindow = 25  # 时间窗口大小，表示每次提取特征时考虑的样本数量（1秒，50Hz采样）
strideWindow = 10  # 步长，表示每次滑动窗口的间隔（无重叠）

# 计算可以提取多少个特征窗口
length = (df.shape[0] - timeWindow) // strideWindow + 1  # 每个类别的片段数量

# 手动输入标签
# 假设所有标签都是
manual_labels = [1] * length  # 为每个窗口分配标签

# 遍历每个特征窗口
for j in range(length):
    # 提取当前窗口的IMU数据
    window_data = df[j * strideWindow:j * strideWindow + timeWindow, :]

    # 提取多种特征
    rms = featureRMS(window_data)  # 均方根值（Root Mean Square）
    mav = featureMAV(window_data)  # 平均绝对值（Mean Absolute Value）
    wl = featureWL(window_data)  # 波形长度（Waveform Length）
    zc = featureZC(window_data)  # 过零率（Zero Crossing）
    ssc = featureSSC(window_data)  # 斜率符号变化（Slope Sign Change）

    # 将所有特征合并为一个特征向量
    featureStack = np.hstack((rms, mav, wl, zc, ssc))

    # 将特征向量添加到特征数据列表中
    featureData.append(featureStack)

    # 添加手动输入的标签
    labels.append(manual_labels[j])

# 将特征数据和标签列表转换为 NumPy 数组
featureData = np.array(featureData)  # 特征数据数组
labels = np.array(labels)  # 标签数组

print("Feature data shape:", featureData.shape)  # 打印特征数据的形状
print("Labels shape:", labels.shape)  # 打印标签数组的形状

# 保存特征数据和标签到 .npy 文件
np.save('DeepDunfeatureData.npy', featureData)
np.save('DeepDunlabels.npy', labels)
print(f"特征数据已保存到文件: DeepDunfeatureData.npy")
print(f"标签数据已保存到文件: DeepDunlabels.npy")