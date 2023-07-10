import os
import sys
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np

from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.time_delta import TimeDelta
from algorithms.features.impl.pname_ret import ProcessNameAndRet
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.decision_engines.scg_gp import SystemCallGraph
from algorithms.features.impl.return_value import ReturnValue
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.w2v_embedding import W2VEmbedding

from dataloader.dataloader_factory import dataloader_factory
from dataloader.data_loader_2021 import RecordingType
from dataloader.direction import Direction


DATAOUT_DIR='K:/hids/dataSet/GP_DATA_DIR'

def create_scg(dataloader, scenario_name):
    save_file_path = os.path.join(DATAOUT_DIR, scenario_name)
    out_dir = os.path.join(save_file_path, 'SCG')
    if  os.path.exists(save_file_path) is not True:
        os.mkdir(save_file_path)

    if os.path.exists(out_dir) is not True:
            os.mkdir(out_dir)

    intEmbed = IntEmbedding(scenario_name = scenario_name)
    pnr = ProcessNameAndRet(intEmbed)
    sc = SystemCallGraph(pnr)

    if intEmbed.need_train:
        for recording in tqdm(dataloader.training_data(),
                                  f"Handle {scenario_name} Train".rjust(27),
                                  unit=" recording_train"):

            for syscall in recording.syscalls():
                intEmbed.train_on(syscall)

        intEmbed.fit()

    print('intEmbed train End')
    for recording in tqdm(dataloader.training_data(),
                                  f"Handle {scenario_name} Train".rjust(27),
                                  unit=" recording_train"):
        for syscall in recording.syscalls():
            sc.train_on(syscall)

    nx.write_multiline_adjlist(sc._graphs['mysqld'], os.path.join(out_dir, 'mysqld_scg'), delimiter='|')
    nx.write_multiline_adjlist(sc._graphs['apache2'], os.path.join(out_dir, 'apache2_scg'), delimiter='|')


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

    # getting the LID-DS base path from argument or environment variable
    work_dir = os.path.join('K:/hids', "dataSet")
    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(work_dir, scenario_name)
        # 如果 GP_DATA_DIR 目录下不存在 scenario_name 目录
        # if GP_DATA_DIR path do not exist scenario_name dir
        if not os.path.exists(os.path.join(DATAOUT_DIR, scenario_name)):
            print('scenario_path not exist, Create')
            os.mkdir(os.path.join(DATAOUT_DIR, scenario_name))

        intEmbed = IntEmbedding(scenario_name=scenario_name)
        pnr = ProcessNameAndRet(intEmbed)

        dataloader = dataloader_factory(scenario_path, direction=Direction.BOTH)

        # create_scg(dataloader, scenario_name)
        # features
        # 'time','UserID', 'PID', 'ProcessName', 'TID', 'syscall', 'DIR', 'ARGS'
        # 1. 每秒 时间间隔 数量， 针对每一项进行频率统计
        # 2. PID and TID switch frequency
        # 3. syscall call graph create, user intEmbedding + return value(default 0)
        # TODO: ARGs Analysis
        ##################

        for recording in tqdm(dataloader.training_data(),
                              f"Handle {scenario_name} Train".rjust(27),
                              unit=" recording_train"):

            _graphs = {}
            _last_added_nodes = {}
            _graphs['mysqld'] = nx.DiGraph()
            _graphs['apache2'] = nx.DiGraph()
            scg = SystemCallGraph(pnr)
            df = pd.DataFrame(columns=['time', 'PID', 'ProcessName', 'TID', 'syscallInt', 'Ret'])
            # 每一项代表1 ms, 最后一项代表大于 31 ms
            time_inter = [0] * 32
            recording_type = dataloader._metadata_list['training'][recording.name]['recording_type']
            if recording_type == RecordingType.NORMAL_AND_ATTACK or recording_type == RecordingType.ATTACK:
                print(f'Name {recording_type.name} is Traing Data, but has Attack file, Check')

            recoding_path = os.path.join(DATAOUT_DIR, scenario_name, recording.name)
            data_path = os.path.join(recoding_path, 'FinalData')
            dataframe_path = os.path.join(recoding_path, 'DataFrame')
            scg_path = os.path.join(recoding_path, 'SCG')
            if not os.path.exists(recoding_path):
                os.mkdir(recoding_path)
                os.mkdir(dataframe_path)
                os.mkdir(scg_path)
                os.mkdir(data_path)

            for syscall in recording.syscalls():
                # prepare time interval
                syscall_time = syscall.timestamp_unix_in_ns()
                pname = syscall.process_name()
                tid = syscall.thread_id()
                # put thing to queue
                syscallARet = pnr._calculate(syscall)

                df.loc[len(df.index)] = [syscall_time, syscall.process_id(), pname, tid, syscallARet[0], syscallARet[1]]

                scg.train_on(syscall)

                # 1 S 时间到了，进行处理
                if syscall_time - df.at[0, 'time'] > 1000000000:
                    df.to_pickle(os.path.join(dataframe_path, str(df.at[0, 'time']) + '.pkl'))
                    data = []
                    # 1s handle
                    # 创建区分时间的箱
                    timedel_bin = [i for i in range(0, 32)]
                    timedel_bin.extend([i for i in range(32, 256, 32)])
                    # 获取时间间隔，去掉 nan
                    handle_df = (df['time'].diff() / (10 ** 6)).dropna(axis=0, how='all')
                    # 按照箱 bin 切分时间间隔
                    category_count = pd.cut(x=handle_df, bins=timedel_bin)
                    # 统计 nan 值， nan 值为大于范围的，统一放在最后
                    most_time_cnt = category_count.isna().sum()
                    data.extend(category_count.value_counts(sort=False).to_list())
                    # add most_time_cnt
                    data.append(most_time_cnt)

                    # 统计 PID 切换频率
                    pid_sw_fre = len(df[df['PID'].diff() != 0])
                    pid_num = len(df['PID'].value_counts())
                    data.append(pid_sw_fre)
                    data.append(pid_num)
                    # 统计 TID 切换频率
                    tid_sw_fre = len(df[df['TID'].diff() != 0])
                    tid_num = len(df['TID'].value_counts())
                    data.append(tid_sw_fre)
                    data.append(tid_num)

                    np.save(os.path.join(data_path, str(df.at[0, 'time']) + '.npy'), np.array(data))

                    # remove first call from syscall graph
                    nx.write_multiline_adjlist(scg._graphs['mysqld'], os.path.join(scg_path,str(df.at[0, 'time']) + '_mysqld_scg'), delimiter='|')
                    nx.write_multiline_adjlist(scg._graphs['apache2'],
                                               os.path.join(scg_path, str(df.at[0, 'time']) + '_apache2_scg'),
                                               delimiter='|')
                    src_node = tuple([df.at[0, 'syscallInt'], df.at[0, 'Ret']])
                    tmp = df[['syscallInt', 'Ret']][df['TID'].isin([df.at[0, 'TID']])]
                    if len(tmp) > 1:
                        tar_node = tuple([tmp.at[tmp.index[1], 'syscallInt'], tmp.at[tmp.index[1], 'Ret']])
                        scg.remove_one_edge(df.at[0, 'ProcessName'], src_node, tar_node)
                    else:
                        print(f'{syscall_time} has the last tid {tid}')

                    # remove df index 0 and reset index
                    df.drop(index=0, inplace=True)
                    df.reset_index(drop=True, inplace=True)

        print('End')
