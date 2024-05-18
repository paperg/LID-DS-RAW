
from pymongo import MongoClient
import os
import sys
# import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (50,30)

SCENARIOS = [
    "EPS_CWE-434",
    "CVE-2020-9484",
    "PHP_CWE-434",
    "CVE-2020-23839",
    "CVE-2019-5418",
    "CVE-2014-0160",
    "Bruteforce_CWE-307",
    "CVE-2017-7529",
    "CWE-89-SQL-injection",
    "CVE-2018-3760",
    "CVE-2012-2122",
    "CVE-2020-13942",
    "Juice-Shop",
]

MODEL_NAME = [
    "CHBS_Model",
    "Mine_final",
    "Mine_No_threshold",
    "Mine_CHBS_Model",
    "Mine_CHBS_threshold",
    "CHBS_Data_Mine_Model",
    # "Mine_no_transformer",
    'Mine_AE_hideen_2',
    'Mine_test_no_trasnformer',
    'Mine_test',
    'Mine_test_AE'
]

DISPLAY_FILED = [
    "false_positives",
    # "true_positives",
    # "true_negatives",
    "false_negatives",
    "detection_rate",
    "consecutive_false_positives_normal",
    "consecutive_false_positives_exploits",
    "precision_with_cfa",
    "precision_with_syscalls",
    "f1_cfa"
]

# {'_id':
# 'false_positives': 0,
# 'true_positives': 340,
# 'true_negatives': 114566,
# 'false_negatives': 3485,
# 'correct_alarm_count': 104,
# 'exploit_count': 120,
# 'detection_rate': 0.8666666666666667,
# 'consecutive_false_positives_normal': 0,
# 'consecutive_false_positives_exploits': 0,
# 'recall': 0.8666666666666667,
# 'precision_with_cfa': 1.0,
# 'precision_with_syscalls': 1.0,
# 'f1_cfa': 0.9285714285714286,
# 'scenario': 'CVE-2019-5418',
# 'Model': 'CHBS_Model',
# 'date': '2024-01-12',
# 'detection_time': 36.04842443068822
# }
collection_name = 'experiments'
client = MongoClient()
collection = client[collection_name][collection_name]

y_result_dict = {}

for filed in DISPLAY_FILED:
    y_result_dict[filed] = {}
    for scenario in SCENARIOS:
        y_result_dict[filed][scenario] = {}
        for model in MODEL_NAME:
            for content in collection.find({ 'scenario': scenario, 'Model': model}):
                print(content)
                y_result_dict[filed][scenario][model] = content[filed]

columns = ['scenario']
for filed in DISPLAY_FILED:
    for model in MODEL_NAME:
        columns.append(filed + '_' + model)

df = pd.DataFrame(columns=columns)

for scenario in SCENARIOS:
    row_data = [scenario]
    for i, filed in enumerate(DISPLAY_FILED):
        for j, model in enumerate(MODEL_NAME):
            row_data.append(y_result_dict[filed][scenario][model])
    df.loc[len(df)] = row_data

df.to_csv('./result.csv')

# TO Json
# data = []
# for document in collection.find():
#     data.append(document)
# json_data = json.dumps(data, indent=4, default=str)
# print(json_data)