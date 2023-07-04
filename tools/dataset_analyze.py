
import os
import sys
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from rich.table import Table
from rich import print as rptint

from dataloader.direction import Direction
from dataloader.dataloader_factory import dataloader_factory
from dataloader.data_loader_2021_df import DF_DataLoader2021
from dataloader.data_loader_2021 import RecordingType
from dataloader.data_loader_2021_df import RecordingType as DF_RecordingType

NORMAL = 'NORMAL'
NORMAL_AND_ATTACK = 'NORMAL_AND_ATTACK'
ATTACK = 'ATTACK'
IDLE = 'IDLE'

recording_type_str = {RecordingType.NORMAL:NORMAL, RecordingType.NORMAL_AND_ATTACK:NORMAL_AND_ATTACK, RecordingType.ATTACK:ATTACK, RecordingType.IDLE:IDLE}

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def display_file(train_file_dict, val_file_dict, test_file_dict):
    output_table = Table(title='File Statistics')
    output_table.add_column("DIR", style="magenta")
    output_table.add_column("NORMAL", style="magenta")
    output_table.add_column("NORMAL_AND_ATTACK", style="magenta")
    output_table.add_column("ATTACK", style="magenta")
    output_table.add_column("IDLE", style="magenta")

    output_table.add_row('Trainning', str(train_file_dict[RecordingType.NORMAL]), str(train_file_dict[RecordingType.NORMAL_AND_ATTACK]),
                         str(train_file_dict[RecordingType.ATTACK]), str(train_file_dict[RecordingType.IDLE]))
    output_table.add_row('Validation', str(val_file_dict[RecordingType.NORMAL]), str(val_file_dict[RecordingType.NORMAL_AND_ATTACK]),
                         str(val_file_dict[RecordingType.ATTACK]), str(val_file_dict[RecordingType.IDLE]))
    output_table.add_row('Test', str(test_file_dict[RecordingType.NORMAL]), str(test_file_dict[RecordingType.NORMAL_AND_ATTACK]),
                         str(test_file_dict[RecordingType.ATTACK]), str(test_file_dict[RecordingType.IDLE]))

    rptint(output_table)

# 统计文件个数与类型
def get_file_num(dataloader):
    # print(dataloader._metadata_list)
    # 统计文件个数
    # Trainning
    train_file_dict = {RecordingType.NORMAL: 0, RecordingType.NORMAL_AND_ATTACK: 0, RecordingType.ATTACK: 0,
                       RecordingType.IDLE: 0}
    file_list = sorted(dataloader._metadata_list['training'].keys())
    for file in file_list:
        t = dataloader._metadata_list['training'][file]['recording_type']
        train_file_dict[t] = train_file_dict[t] + 1

    # validation
    val_file_dict = {RecordingType.NORMAL: 0, RecordingType.NORMAL_AND_ATTACK: 0, RecordingType.ATTACK: 0,
                     RecordingType.IDLE: 0}
    file_list = sorted(dataloader._metadata_list['validation'].keys())
    for file in file_list:
        t = dataloader._metadata_list['validation'][file]['recording_type']
        val_file_dict[t] = val_file_dict[t] + 1
    # Test
    test_file_dict = {RecordingType.NORMAL: 0, RecordingType.NORMAL_AND_ATTACK: 0, RecordingType.ATTACK: 0,
                      RecordingType.IDLE: 0}
    file_list = sorted(dataloader._metadata_list['test'].keys())
    for file in file_list:
        t = dataloader._metadata_list['test'][file]['recording_type']
        test_file_dict[t] = test_file_dict[t] + 1

    display_file(train_file_dict, val_file_dict, test_file_dict)

def display_syscall_num(train_syscall_num, val_syscall_num, test_syscall_num, test_sc_num_nor_attack, Save_File:bool):

    # 系统调用名称 个数
    output_table = Table(title='System Call Number Statistics')
    output_table.add_column("DIR", style="magenta")
    output_table.add_column("NORMAL", style="magenta")
    output_table.add_column("NORMAL_AND_ATTACK", style="magenta")
    output_table.add_column("ATTACK", style="magenta")
    output_table.add_column("IDLE", style="magenta")

    output_table.add_row('Trainning', str(len(train_syscall_num[NORMAL])), str(len(train_syscall_num[NORMAL_AND_ATTACK])),
                         str(len(train_syscall_num[ATTACK])), str(len(train_syscall_num[IDLE])))
    output_table.add_row('Validation', str(len(val_syscall_num[NORMAL])), str(len(val_syscall_num[NORMAL_AND_ATTACK])),
                         str(len(val_syscall_num[ATTACK])), str(len(val_syscall_num[IDLE])))
    output_table.add_row('Test', str(len(test_syscall_num[NORMAL])), str(len(test_syscall_num[NORMAL_AND_ATTACK])),
                         str(len(test_syscall_num[ATTACK])), str(len(test_syscall_num[IDLE])))

    rptint(output_table)

    if Save_File:
        with open('./out/train_syscall_num', 'w') as f:
            json.dump(train_syscall_num, f, indent=4, ensure_ascii=True, sort_keys=False)

        with open('./out/val_syscall_num', 'w') as f:
            json.dump(val_syscall_num, f, indent=4, ensure_ascii=True, sort_keys=False)

        with open('./out/test_syscall_num', 'w') as f:
            json.dump(test_syscall_num, f, indent=4, ensure_ascii=True, sort_keys=False)

        with open('./out/test_sc_num_nor_attack', 'w') as f:
            json.dump(test_sc_num_nor_attack, f, indent=4, ensure_ascii=True, sort_keys=False)

    # 系统调用未见个数，val和 test相比较与 train
    # ATTACK 比 NORMAL
    a = test_syscall_num[NORMAL].keys() - train_syscall_num[NORMAL].keys()
    print(f'Test Normal : Trainning Normal {len(a)} : {a}')

    a = test_syscall_num[NORMAL_AND_ATTACK].keys() - train_syscall_num[NORMAL].keys()
    print(f'Test Normal and attack : Trainning Normal {len(a)} : {a}')

    a = test_syscall_num[ATTACK].keys() - train_syscall_num[NORMAL].keys()
    print(f'Test attack : Trainning Normal {len(a)} : {a}')

    print(f'Test Normal SC Num {len(test_sc_num_nor_attack[NORMAL].keys())}, Attack SC Num {len(test_sc_num_nor_attack[ATTACK].keys())}')
    a = test_sc_num_nor_attack[ATTACK].keys() - test_sc_num_nor_attack[NORMAL].keys()
    print(f'Test Seq attack : Test Normal {len(a)} : {a}')

    a = test_sc_num_nor_attack[NORMAL].keys() - test_sc_num_nor_attack[ATTACK].keys()
    print(f'Test Seq Normal : Test attack {len(a)} : {a}')

def analyze_syscall_obj(dataloader, scenario_name):
    TIME_PERIOD = 1

    save_file_path = os.path.join('./out', scenario_name)
    if  os.path.exists(save_file_path) is not True:
        os.mkdir(save_file_path)

    # 文件中系统调用 字典 {名称， 数量}
    Save_File = True
    if os.path.exists(os.path.join(save_file_path, 'train_syscall_num')):
        print('Dict File Has Exists')
        Save_File = False
        with open(os.path.join(save_file_path, 'train_syscall_num'), 'r') as f:
            train_syscall_num = json.load(f)

        with open(os.path.join(save_file_path, 'test_syscall_num'), 'r') as f:
            test_syscall_num = json.load(f)

        with open(os.path.join(save_file_path, 'val_syscall_num'), 'r') as f:
            val_syscall_num = json.load(f)
    else:
    #     train_syscall_num = {NORMAL: {}, NORMAL_AND_ATTACK: {}, ATTACK: {},
    #                          IDLE: {}}
    #     train_sc_num_persec = {NORMAL: [], NORMAL_AND_ATTACK: [], ATTACK: [],
    #                          IDLE: []}
    #     for recording in tqdm(dataloader.training_data(),
    #                           f"Handle {scenario_name} Train".rjust(27),
    #                           unit=" recording_train"):
    #         recording_type = recording_type_str[dataloader._metadata_list['training'][recording.name]['recording_type']]
    #         start_time = recording.metadata()["time"]["warmup_end"]["absolute"] * (10 ** 9)
    #         dict_persec = {}
    #         for syscall in recording.syscalls():
    #             """
    #             Read json file and extract metadata as dict
    #             with following format:
    #             {"container": [
    #                "ip": str,
    #                "name": str,
    #                "role": str
    #             "exploit": bool,
    #             "exploit_name": str,
    #             "image": str,
    #             "recording_time": int,
    #             "time":{
    #                "container_ready": {
    #                    "absolute": float,
    #                    "source": str
    #                },
    #                "exploit": [
    #                    {
    #                        "absolute": float,
    #                        "name": str,
    #                        "source": str
    #                    }
    #                ]
    #                "warmup_end": {
    #                    "absolute": float,
    #                    "source": str
    #                }
    #             }
    #             }
    #
    #             Returns:
    #             dict: metadata dictionary
    #             self._current_exploit_time = recording.metadata()["time"]["exploit"][0]["absolute"]
    #             """
    #             # 系统调用序列
    #             train_syscall_num[recording_type][syscall.name()] = train_syscall_num[recording_type].get(syscall.name(), 0) + 1
    #
    #             dict_persec[syscall.name()] = dict_persec.get(syscall.name(), 0) + 1
    #             if syscall.timestamp_unix_in_ns() - start_time > 1 * 1000 * 1000 * 1000:
    #                 start_time = syscall.timestamp_unix_in_ns()
    #                 train_sc_num_persec[recording_type].append(dict_persec)
    #                 dict_persec = {}
    #
    #
    #
    #     val_syscall_num = {NORMAL: {}, NORMAL_AND_ATTACK: {}, ATTACK: {},
    #                          IDLE: {}}
    #
    #     for recording in tqdm(dataloader.validation_data(),
    #                           f"Handle {scenario_name} Validation".rjust(27),
    #                           unit=" recording_val"):
    #         recording_type = recording_type_str[dataloader._metadata_list['validation'][recording.name]['recording_type']]
    #         for syscall in recording.syscalls():
    #             val_syscall_num[recording_type][syscall.name()] = val_syscall_num[recording_type].get(syscall.name(), 0) + 1

        # 对整体文件进行评估
        test_syscall_num = {NORMAL: {}, NORMAL_AND_ATTACK: {}, ATTACK: {},
                             IDLE: {}}
        # 对异常文件，通过 explot time 进行分割，评估
        test_sc_num_nor_attack = {NORMAL:{}, ATTACK:{}}

        # time variable analysis
        # 每秒，系统调用调用次数
        test_sc_num_persec = {NORMAL: [], ATTACK: []}
        # 记录每秒，间隔时间分布
        test_sc_time_interval_persec = {NORMAL:[], ATTACK:[]}

        for recording in tqdm(dataloader.test_data(),
                              f"Handle {scenario_name} Test".rjust(27),
                              unit=" recording_test"):
            recording_type = recording_type_str[dataloader._metadata_list['test'][recording.name]['recording_type']]
            exploit_start_time = 0
            recoding_start_time = 0
            # 每秒系统调用个数
            dict_sc_persec = {}
            # 每秒间隔时间计数  20ms and over 20 ms
            inter_persec = [0] * 21
            previous_time = 0

            if recording_type == RecordingType.ATTACK or recording_type == NORMAL_AND_ATTACK:
                exploit_start_time = recording.metadata()["time"]["exploit"][0]["absolute"] * (10 ** 9)
                # print(f'Attack file , exploit_start_time  {exploit_start_time}\n')
                if exploit_start_time == 0:
                    print('!!!!!!!!!!!!! exploit_start_time! why it is 0')

            for syscall in recording.syscalls():
                # record time interval
                if recoding_start_time == 0:
                    recoding_start_time = syscall.timestamp_unix_in_ns()
                    previous_time = recoding_start_time
                # calcaulate time interval
                inreval = syscall.timestamp_unix_in_ns() - previous_time
                previous_time = syscall.timestamp_unix_in_ns()
                pos = int(inreval // (10 ** 6))
                if pos > 20:
                    pos = 20

                inter_persec[pos] += 1

                dict_sc_persec[syscall.name()] = dict_sc_persec.get(syscall.name(), 0) + 1
                if syscall.timestamp_unix_in_ns() - recoding_start_time > TIME_PERIOD * 1000 * 1000 * 1000:
                    recoding_start_time = syscall.timestamp_unix_in_ns()
                    if exploit_start_time == 0 or recoding_start_time < exploit_start_time:
                        test_sc_num_persec[NORMAL].append(dict_sc_persec)
                        test_sc_time_interval_persec[NORMAL].append(inter_persec)
                    else:
                        test_sc_num_persec[ATTACK].append(dict_sc_persec)
                        test_sc_time_interval_persec[ATTACK].append(inter_persec)

                    dict_sc_persec = {}
                    inter_persec = [0] * 21

                # record NORMAL and ATTACK syscall num detail in test file
                test_syscall_num[recording_type][syscall.name()] = test_syscall_num[recording_type].get(syscall.name(), 0) + 1
                if exploit_start_time == 0:
                    test_sc_num_nor_attack[NORMAL][syscall.name()] = test_sc_num_nor_attack[NORMAL].get(syscall.name(), 0) + 1
                else:
                    if syscall.timestamp_unix_in_ns() < exploit_start_time:
                        test_sc_num_nor_attack[NORMAL][syscall.name()] = test_sc_num_nor_attack[NORMAL].get(
                            syscall.name(), 0) + 1
                    else:
                        test_sc_num_nor_attack[ATTACK][syscall.name()] = test_sc_num_nor_attack[NORMAL].get(
                            syscall.name(), 0) + 1
                # end

    # print(recording.metadata())
    # display_syscall_num(train_syscall_num, val_syscall_num, test_syscall_num, test_sc_num_nor_attack, Save_File)
    with open(os.path.join(save_file_path, 'test_sc_num_persec_' + str(TIME_PERIOD)), 'w') as f:
        json.dump(test_sc_num_persec, f, indent=4, ensure_ascii=True, sort_keys=False)

    with open(os.path.join(save_file_path, 'test_sc_time_interval_persec_' + str(TIME_PERIOD)), 'w') as f:
        json.dump(test_sc_time_interval_persec, f, indent=4, ensure_ascii=True, sort_keys=False)

    with open(os.path.join(save_file_path, 'test_sc_num_nor_attack'), 'w') as f:
        json.dump(test_sc_num_nor_attack, f, indent=4, ensure_ascii=True, sort_keys=False)

    print('End')

def get_dataframe(dataloader, scenario_name):
    save_file_path = os.path.join('./out', scenario_name)
    if  os.path.exists(save_file_path) is not True:
        os.mkdir(save_file_path)
        os.mkdir(os.path.join(save_file_path, 'DataFrame'))

    test_normal_time_df = pd.Series()
    test_exploit_time_df = pd.Series()

    test_normal_df = pd.DataFrame()
    test_exploit_df = pd.DataFrame()

    file_record = {}
    try:
        for recording in tqdm(dataloader.test_data(),
                                  f"Handle {scenario_name} Train".rjust(27),
                                  unit=" recording_train"):
            exploit_start_time = 0
            recording_type = dataloader._metadata_list['test'][recording.name]['recording_type']
            if recording_type == DF_RecordingType.NORMAL_AND_ATTACK or recording_type == DF_RecordingType.ATTACK:
                exploit_start_time = recording.metadata()["time"]["exploit"][0]["absolute"] * (10 ** 9)
                # print(f'Attack file , exploit_start_time  {exploit_start_time}\n')
                if exploit_start_time == 0:
                    print('!!!!!!!!!!!!! exploit_start_time! why it is 0')

            for sc_df, name in recording.syscalls():
                if sc_df.index[0] != 0:
                    print('!!!!!!!!!!!!! sc_df.index[0] != 0, Skip')
                    print(name)
                    print(sc_df.index[0])
                    continue

                if exploit_start_time != 0:
                    file_record[name + '.pkl'] = exploit_start_time

                sc_df.to_pickle(os.path.join(save_file_path, 'DataFrame', name + '.pkl'))
                if exploit_start_time != 0:
                    sc_df_nor = sc_df[sc_df['time'] < exploit_start_time]
                    sc_df_exp = sc_df[sc_df['time'] >= exploit_start_time]

                    time = sc_df_nor['time']
                    time_next = time[1:].reset_index(drop=True)
                    df = time_next - time
                    test_normal_time_df = test_normal_time_df.append(df, ignore_index =True)
                    test_normal_df = test_normal_df.append(sc_df_nor, ignore_index=True)

                    time = sc_df_exp['time'].reset_index(drop=True)
                    time_next = time[1:].reset_index(drop=True)
                    df = time_next - time
                    test_exploit_time_df = test_exploit_time_df.append(df, ignore_index =True)
                    test_exploit_df = test_exploit_df.append(sc_df_exp, ignore_index=True)

                else:
                    time = sc_df['time']
                    time_next = time[1:].reset_index(drop=True)
                    df = time_next - time
                    test_exploit_time_df = test_exploit_time_df.append(df, ignore_index =True)

                    test_normal_df = test_normal_df.append(sc_df, ignore_index=True)

        test_normal_time_df.to_pickle(os.path.join(save_file_path,  'DF_test_normal_time.pkl'))
        test_exploit_time_df.to_pickle(os.path.join(save_file_path, 'DF_test_exploit_time.pkl'))

        test_normal_df.to_pickle(os.path.join(save_file_path,  'DF_test_normal.pkl'))
        test_exploit_df.to_pickle(os.path.join(save_file_path, 'DF_test_exploit.pkl'))

        with open(os.path.join(save_file_path, scenario_name + '.json'), 'w') as f:
            json.dump(file_record, f, indent=4, ensure_ascii=True, sort_keys=False)

    except:
        print(name)
        print(sc_df)

if __name__ == '__main__':

    LID_DS_VERSION_NUMBER = 1

    LID_DS_VERSION = [
            "LID-DS-2019",
            "LID-DS-2021"
            ]
    # scenarios ordered by training data size asc
    SCENARIOS = [
      "CWE-89-SQL-injection",
      "CVE-2017-7529",
      "CVE-2014-0160",
      "CVE-2012-2122",
      "Bruteforce_CWE-307",
      "CVE-2020-23839",
      "PHP_CWE-434",
      "ZipSlip",
      "CVE-2018-3760",
      "CVE-2020-9484",
      "EPS_CWE-434",
      "CVE-2019-5418",
      "Juice-Shop",
      "CVE-2020-13942",
      "CVE-2017-12635_6"
    ]
    SCENARIO_RANGE = SCENARIOS[0:1]

    LID_DS_BASE_PATH = 'K:/hids'

    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(LID_DS_BASE_PATH,
                                     "dataSet",
                                     scenario_name)
        dataloader = dataloader_factory(scenario_path, direction=Direction.BOTH)
        dataloader_df =  DF_DataLoader2021(scenario_path, direction=Direction.BOTH)
        '''
            class RecordingType(Enum):
                NORMAL = 1
                NORMAL_AND_ATTACK = 2
                ATTACK = 3
                IDLE = 4
            temp_dict = {
                'recording_type': recording_type,
                'path': file
            }
            metadata_dict[TRAINING][get_file_name(file)] = temp_dict
            TRAINING = 'training'
            VALIDATION = 'validation'
            TEST = 'test'
        '''
        # 统计文件个数
        get_file_num(dataloader)
        # analyze_syscall_obj(dataloader, scenario_name)
        get_dataframe(dataloader_df, scenario_name)

        print('Success')