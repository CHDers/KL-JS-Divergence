# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2023/5/25  22:13
# @Author: Yanjun Hao
# @File  : KL_JS.py


import numpy as np
import pandas as pd
from scipy.special import kl_div
from scipy.stats import entropy
import scipy
from rich import print


def read_data(filepath: str, column_name: str) -> list:
    df = pd.read_csv(filepath)
    return df[column_name].values.tolist()


def data_process(data1: list, data2: list) -> object:
    """
    将两个不同长度的list按照list长度的最小值转为相同长度
    :param data1:
    :param data2:
    :return:
    """
    if len(data1) != len(data2):
        max_length = min(len(data1), len(data2))
        new_data1 = np.array(data1[:max_length])
        new_data2 = np.array(data2[:max_length])
    else:
        new_data1 = np.array(data1)
        new_data2 = np.array(data2)
    return new_data1, new_data2


def get_kl_divergence_1(data1: list, data2: list) -> float:
    """
    Linking: https://blog.csdn.net/qq_27782503/article/details/121830753
    :param data1:
    :param data2:
    :return:
    """
    new_data1, new_data2 = data_process(data1=data1, data2=data2)
    new_data1 = new_data1 / np.sum(new_data1)
    new_data2 = new_data2 / np.sum(new_data2)
    kl_divergence = kl_div(new_data1, new_data2).sum()
    return kl_divergence


def get_kl_divergence_2(data1: list, data2: list) -> float:
    """
    计算离散变量之间的KL散度
    Linking: https://blog.csdn.net/qq_27782503/article/details/121830753
    :param data1:
    :param data2:
    :return:
    """
    new_data1, new_data2 = data_process(data1=data1, data2=data2)
    kl_divergence = scipy.stats.entropy(new_data1, new_data2)
    return kl_divergence


def get_js_divergence(data1: list, data2: list) -> float:
    """
    计算离散变量之间的JS散度
    Linking: https://blog.csdn.net/blmoistawinde/article/details/84329103
    :param data1:
    :param data2:
    :return:
    """
    new_data1, new_data2 = data_process(data1=data1, data2=data2)
    M = 0.5 * (new_data1 + new_data2)
    js_divergence = 0.5 * (entropy(new_data1, M) + entropy(new_data2, M))
    return js_divergence


if __name__ == '__main__':
    # 两组离散数据
    p = read_data(filepath="./datasets/raw_50k.csv", column_name="srcip")
    q = read_data(filepath="./datasets/output_50k.csv", column_name="srcip")
    # p = [0.1, 0.2, 0.3, 0.4]
    # q = [0.3, 0.2, 0.15, 0.35]
    # p = [1, 2, 3, 4]
    # q = [3, 2, 15, 35]
    print(p[:10], q[:10])
    kl_divergence_1 = get_kl_divergence_1(data1=p, data2=q)
    kl_divergence_2 = get_kl_divergence_2(data1=p, data2=q)
    kl_divergence_2_reverse = get_kl_divergence_2(data1=q, data2=p)
    js_divergence = get_js_divergence(data1=p, data2=q)
    js_divergence_reverse = get_js_divergence(data1=q, data2=p)
    print(f"kl_divergence_1={kl_divergence_1},"
          f"kl_divergence_2={kl_divergence_2},"
          f"kl_divergence_2_reverse={kl_divergence_2_reverse},"
          f"js_divergence={js_divergence},"
          f"js_divergence_reverse={js_divergence_reverse}"
          )
