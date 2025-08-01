import joblib
import pandas as pd
import numpy as np
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf

from data_features import *
import sklearn


def predict_with_tflite(model_path, X_data, n_steps=10, n_signals=18):
    """使用TensorFlow Lite模型进行预测"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    X_reshaped = X_data.reshape((X_data.shape[0], n_steps, n_signals))

    predictions = []

    # 逐个样本进行预测，避免批次大小不匹配问题
    for i in range(X_reshaped.shape[0]):
        sample = X_reshaped[i:i + 1].astype(np.float32)  # 形状为 [1, n_steps, n_signals]

        # 设置输入张量
        interpreter.set_tensor(input_details[0]['index'], sample)

        # 运行推理
        interpreter.invoke()

        # 获取输出张量
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])

        if (i + 1) % 10 == 0:
            print(f"已预测 {i + 1}/{len(X_reshaped)} 个样本")

    return np.array(predictions)


def normalize_data(X):
    """直接对数据进行归一化处理"""
    # 计算当前批次数据的最小值和最大值
    data_min = np.min(X, axis=0)
    data_max = np.max(X, axis=0)

    # 防止除零错误
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0

    # 归一化到 [0, 1] 范围
    return (X - data_min) / data_range

# 读取数据时处理格式问题
try:
    # 显式指定列名（根据实际数据特征）
    column_names = [
        'timestamp', 'sensor_name',
        'ax', 'ay', 'az', 'qw', 'qx', 'qy', 'qz'
    ]

    # 使用正则表达式分割（处理末尾多余的逗号）
    df = pd.read_csv('rawdata/temp.csv',
                     header=None,
                     skiprows=1,  # 跳过原header
                     sep=r',\s*',  # 匹配逗号+任意空格
                     engine='python',
                     names=column_names)

    # 转换为数值类型
    numeric_cols = column_names[2:]  # 从第三列开始都是数值
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 清除无效数据
    df = df.dropna().reset_index(drop=True)

except Exception as e:
    print(f"数据读取错误: {e}")
    exit()


# 单位转换函数
def convert_units(data):
    # 角速度: deg/s → rad/s (第6-8列)
    gyro_cols = ['qx', 'qy', 'qz']
    data[gyro_cols] = data[gyro_cols] * math.pi / 180

    # 重命名转换后的列
    data = data.rename(columns={
        'qx': '角速度X(rad/s)',
        'qy': '角速度Y(rad/s)',
        'qz': '角速度Z(rad/s)',
    })
    return data


# 执行单位转换
df = convert_units(df)


# 数据处理流程
def process_data(df):
    # 按时间排序
    try:
        # 转换时间列为datetime类型 (使用ISO8601格式)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    except Exception as e:
        print(f"时间格式转换错误: {e}")
        # 尝试自动推断格式
        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)

    df_sorted = df.sort_values(by='timestamp')

    # 按设备分组
    groups = df_sorted.groupby('sensor_name', sort=False)

    # 降采样处理
    processed = []
    for name, group in groups:
        try:
            # 降采样到50Hz（假设原始为100Hz）
            downsampled = group.iloc[::2]  # 实际减采样
            processed.append(downsampled)
        except Exception as e:
            print(f"设备 {name} 降采样失败: {e}")
            continue

    # 检查是否有成功处理的数据
    if not processed:
        raise ValueError("所有设备处理失败，没有有效数据")

    # 合并数据并重置索引
    final_df = pd.concat(processed).reset_index(drop=True)

    # 先按设备号排序，再按时间排序
    final_df = final_df.sort_values(by=['sensor_name', 'timestamp']).reset_index(drop=True)

    # 选择需要的列
    keep_cols = ['sensor_name', 'ax', 'ay', 'az',
                 '角速度X(rad/s)', '角速度Y(rad/s)', '角速度Z(rad/s)']
    return final_df[keep_cols]


# 执行处理流程
try:
    df = process_data(df)

    # 按设备名分组（保持原始顺序）
    groups = df.groupby('sensor_name', sort=False)
    group_names = list(groups.groups.keys())

    # 检查是否存在至少一个分组
    if not group_names:
        raise ValueError("CSV文件中未找到任何设备分组")

    # 获取所有设备组中的最小行数
    min_rows = min(len(groups.get_group(name)) for name in group_names)

    # 初始化结果列表
    final_result = []

    # 按行索引遍历所有有效行（基于最小行数）
    for idx in range(min_rows):
        row_data = []

        # 遍历每个设备组
        for name in group_names:
            group_df = groups.get_group(name)
            # 提取当前行的数据（从第二列开始，共6列）
            row_data.extend(group_df.iloc[idx, 1:7].tolist())

        final_result.append(row_data)

    # 创建DataFrame并保存（无表头、无索引）
    pd.DataFrame(final_result).to_csv('data/test.csv', index=False, header=False)

    print(f"文件已成功保存，有效行数：{min_rows}")

except Exception as e:
    print(f"处理过程中发生错误: {e}")
    exit(1)

# 初始化特征数据列表
featureData = []  # 用于存储所有特征数据
timeWindow = 10  # 时间窗口大小，表示每次提取特征时考虑的样本数量（1秒，50Hz采样）
strideWindow = 8  # 步长，表示每次滑动窗口的间隔（无重叠）

# 假设文件名格式为 '3_1.csv', '3_2.csv', ..., '3_n.csv'
file_list = [
    'data/test.csv',  # 文件 3.csv
    # '3_2.csv',  # 文件 3_2.csv
    # '3_3.csv',  # 文件 3_3.csv
    # 根据需要继续添加更多文件
]

# 遍历每个文件
for file_name in file_list:
    # 读取当前文件
    df = pd.read_csv(file_name, header=0)
    df = df.values

    # 计算可以提取多少个特征窗口
    length = (df.shape[0] - timeWindow) // strideWindow + 1  # 每个文件的片段数量

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

# 将特征数据列表转换为 NumPy 数组
featureData = np.array(featureData)  # 特征数据数组

featureData = normalize_data(featureData)

print("Feature data shape:", featureData.shape)  # 打印特征数据的形状



# 示例输入数据
input_data = featureData # 替换为你的实际输入数据
print("一些有用的信息，以了解数据集的形状和归一化情况：")
print("(X 的形状, 每个 X 的均值, 每个 X 的标准差)")
print(input_data.shape, np.mean(input_data), np.std(input_data))
model_path = 'my_model.tflite'
# 使用TFLite模型进行预测
predictions = predict_with_tflite(model_path, featureData)

# 输出预测结果
predicted_classes = np.argmax(predictions, axis=1)
print(f"预测类别分布: {np.bincount(predicted_classes)}")

# 显示前10个预测结果
print("\n前10个预测结果:")
for i in range(min(10, len(predicted_classes))):
    print(f"样本 {i + 1}: 预测类别 = {predicted_classes[i]}, 概率 = {predictions[i][predicted_classes[i]]:.4f}")


