import os
import sys
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from urllib.parse import urlparse
from string import digits
import pickle

# from algorithms.features.impl.syscall_name import SyscallName
# from algorithms.features.impl.int_embedding import IntEmbedding
# from algorithms.features.impl.time_delta import TimeDelta
from algorithms.features.impl.pname_ret import ProcessNameAndRet
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.decision_engines.scg_gp import SystemCallGraph
# from algorithms.features.impl.return_value import ReturnValue
# from algorithms.features.impl.ngram import Ngram
# from algorithms.features.impl.w2v_embedding import W2VEmbedding
# from algorithms.features.impl.unseen_args import UnseenArgs

from dataloader.dataloader_factory import dataloader_factory
from dataloader.data_loader_2021 import RecordingType
from dataloader.direction import Direction

from multiprocessing import Process

work_dir = os.path.join('L:/hids', "dataSet")
DATAOUT_DIR = os.path.join(work_dir, 'GP_DATA_DIR')

TRAINING = 'training'
VALIDATION = 'validation'
TEST = 'test'

index_timedelta_start = 2
index_timdelta_end = 41
index_ptid_start = 41
index_ptid_end = 45
index_uai = 45
index_usi = 46

# ['write', 'mmap', 'brk', 'writev', 'read', 'sendto', 'recvfrom', 'lseek', 'pwrite', 'pread'] 特殊处理
# write : res 成功写入字节数 data=... 数据
# writev : res 成功写入字节数 data=... 数据
# read : res 成功读取字节数 data=... 数据
# readv
# mmap : mmap < res=7FBD92DEA000 vm_size=112392
# brk : brk < res=559ABD3A7000 vm_size=112596
# sendto : res 成功发送字节数 data=... 数据
# recvfrom : res 成功接收字节数 data=... 数据
# lseek : lseek < res=0
# pwrite : res 成功写入字节数 data=... 数据
# pread : res 成功读取字节数 data=... 数据
SYSCALLS_ARGS_RET = ['write', 'writev', 'read', 'mmap', 'brk', 'sendto', 'recvfrom', 'lseek', 'pwrite', 'pread', 'sendfile']


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def df_get_other_feature(row, pnr):
    scint, ret, retReal = pnr._calculate_m(row.syscall, row.RET)
    if row['syscall'] in ['mmap', 'brk']:
        retReal = int(row['ARGS'].split('=')[1])

    file_path = get_filepath(row)

    return scint, ret, retReal, file_path

def get_filepath(row):
    raw_arg = row['ARGS']
    if row['syscall'] == 'open':
        a, file_path = raw_arg.split('=')
        if a != 'name':
            print(f'{a} is not name')
    elif row['syscall'] == 'stat':
        a, file_path = raw_arg.split('=')
        if a != 'path':
            print(f'{a} is not path')
    elif row['syscall'] == 'clone':
        a, file_path = raw_arg.split('=')
        if a != 'exe':
            print(f'{a} is not exe')
    elif row['syscall'] == 'execve':
        a, file_path = raw_arg.split('=')
        if a != 'exe':
            print(f'{a} is not exe')
    else:
        return np.nan

    return process_path(file_path)

def process_path(file_path_i):
    PATH_LENGTH = 3
    if "(" in file_path_i and ")" in file_path_i:
        file_path_between_par = file_path_i[file_path_i.find("(") + 1:file_path_i.find(")")]
        file_path_before_par = file_path_i[:file_path_i.find("(")]
        file_path_max = max([file_path_between_par, file_path_before_par], key=len)

        if file_path_i.count("/") > PATH_LENGTH:
            file_path_min = min([file_path_between_par, file_path_before_par], key=len)
            file_path_i = file_path_max.replace(file_path_min, "")
        else:
            file_path_i = file_path_max

    else:
        if file_path_i.count("/") > PATH_LENGTH:
            full_path = urlparse(file_path_i)
            file_path_i = os.path.dirname(full_path.path)

    processed_file_path = file_path_i.rsplit('/')
    path_last_element = processed_file_path[-1]

    # we clean the path from random characters and numbers
    if path_last_element.islower():
        processed_file_path[-1] = path_last_element.translate({ord(k): None for k in digits})
        processed_file_paths = '/'.join(processed_file_path)
    else:
        processed_file_paths = '/'.join(processed_file_path[:-1])

    return processed_file_paths

class m_SystemCallGraph():
    def __init__(self):
        # internal data
        self._graphs = {}
        self._last_added_nodes = {}

def df_create_scgraph(row, scg: m_SystemCallGraph, only_sc:bool = False):
    """
    adds the current input to the grpah
    """
    if only_sc:
        new_node = str(row.syscallInt)
    else:
        new_node = str((row.syscallInt, row.Ret))

    # check for threads
    # do not use thread id for now
    tid = row.TID
    # graph id
    pname = row.ProcessName
    if pname is np.nan:
        return
    # check for graph
    # create a new graph for every process
    if pname not in scg._graphs:
        scg._graphs[pname] = nx.DiGraph()

    # check for last added node
    if tid not in scg._last_added_nodes:
        scg._last_added_nodes[tid] = None

    # finally add the input
    if scg._last_added_nodes[tid] is None:
        scg._graphs[pname].add_node(new_node)
    else:
        count = 0
        # edge already in graph? then update its freq.
        if scg._graphs[pname].has_edge(scg._last_added_nodes[tid], new_node):
            count = scg._graphs[pname].edges[scg._last_added_nodes[tid], new_node]["f"]
            # print(count)
        count += 1
        scg._graphs[pname].add_edge(scg._last_added_nodes[tid], new_node, f=count)
    scg._last_added_nodes[tid] = new_node


def Init_intEmbed(dataloader, scenario_name, intEmbed):

    if intEmbed.need_train:
        for recording in tqdm(dataloader.training_data(),
                                  f"Handle {scenario_name} IntEmbedding Train".rjust(27),
                                  unit=" recording_train"):

            for syscall in recording.syscalls():
                intEmbed.train_on(syscall)

        intEmbed.fit()
        print('IntEmbed train End')

def  cal_time_delte(df):
    # 创建区分时间的箱
    timedel_bin = [i for i in range(0, 32)]
    timedel_bin.extend([i for i in range(32, 256, 32)])
    # 获取时间间隔，去掉 nan
    handle_df = (df['time'].diff() / (10 ** 2)).dropna(axis=0, how='all')
    # 按照箱 bin 切分时间间隔
    category_count = pd.cut(x=handle_df, bins=timedel_bin)
    # 统计 nan 值， nan 值为大于范围的，统一放在最后
    most_time_cnt = category_count.isna().sum()

    time_bin = category_count.value_counts(sort=False).to_list()
    time_bin.append(most_time_cnt)
    normal_max = sum(time_bin)
    time_bin = [x / normal_max for x in time_bin]
    return time_bin

def cal_pid_tid_switch_frequency(df):
    # 统计 PID 切换频率
    pid_sw_fre = len(df[df['PID'].diff() != 0])
    # PID 数量
    pid_num = len(df['PID'].value_counts())
    # 统计 TID 切换频率
    tid_sw_fre = len(df[df['TID'].diff() != 0])
    tid_num = len(df['TID'].value_counts())

    return [pid_sw_fre, pid_num, tid_sw_fre, tid_num]

def create_scg_for_second(df, scg_path):
    #Create a syscall graph
    scg = m_SystemCallGraph()
    df.apply(df_create_scgraph, args=(scg,), axis=1)

    for pname in scg._graphs.keys():
        nx.write_graphml(scg._graphs[pname],
                                   os.path.join(scg_path, str(df.iloc[-1]['time']) + '_' + pname + '.xml'), encoding='utf-8')
    return scg

def get_uai(sequence_data, seen_args, scg_second, train_scg, exploit_start_time, result_array, save_dir):
    # open : open < fd=13(<f>/etc/apache2/.htpasswd) name=file_name
    # stat  : stat < res=0 path=/var/www/private/upload.php
    # clone : clone < res=466 exe=/usr/sbin/apache2
    # execve:
    uai = 0
    cur_uai = 0
    cur_uai_set = 0
    for pname in sequence_data['ProcessName'].dropna().unique():
        tmp_result = []
        if pname not in seen_args:
            print(f'Process {pname} not in seen_args, Skip')
            continue
        seen_args_second_df = sequence_data[sequence_data['ProcessName'] == pname][['time', 'UserID', 'syscallInt','Ret','Params']].dropna()
        if len(seen_args_second_df) > 0:

            unseen_args = seen_args_second_df.loc[~seen_args_second_df.Params.isin(seen_args[pname])]
            if len(unseen_args) > 0:
                result = {}
                distinct_unseen_args = set(unseen_args['Params'])
                # distinct_syscalls_with_unseen_args = set(unseen_args.apply(lambda X: (X.syscallInt, X.Ret), axis=1))
                distinct_syscalls_with_unseen_args = set(unseen_args.apply(lambda X: X.syscallInt, axis=1))
                in_centrality = []
                out_centrality = []
                train_in_centrality = []
                train_out_centrality = []

                for node in distinct_syscalls_with_unseen_args:
                    node = str(node)
                    uan_in_centrality = nx.in_degree_centrality(scg_second._graphs[pname]).get(node)
                    uan_out_centrality = nx.out_degree_centrality(scg_second._graphs[pname]).get(node)

                    train_uan_in_centrality = nx.in_degree_centrality(train_scg._graphs[pname]).get(node)
                    train_uan_out_centrality = nx.out_degree_centrality(train_scg._graphs[pname]).get(node)

                    if isinstance(uan_in_centrality, type(None)):
                        uan_in_centrality = 0

                    if isinstance(uan_out_centrality, type(None)):
                        uan_out_centrality = 0

                    in_centrality.append(uan_in_centrality)
                    out_centrality.append(uan_out_centrality)
                    train_in_centrality.append(train_uan_in_centrality)
                    train_out_centrality.append(train_uan_out_centrality)

                cur_uai = len(distinct_unseen_args)
                cur_usa = len(unseen_args)
                uai += cur_uai
                # Display
                is_exploit = 0
                if exploit_start_time != 0:
                    if sequence_data.iloc[-1]['time'] > exploit_start_time:
                        is_exploit = 1

                result['Procee Name'] = pname
                result['exploit_start_time'] = exploit_start_time
                result['Current Time'] = float(sequence_data.iloc[-1]['time'])
                result['is_exploit'] = is_exploit
                result['cur_usa'] = cur_usa
                result['cur_uai'] = cur_uai
                result['uai'] = uai
                if uai > 2:
                    print(uai)
                usa_list = unseen_args['Params'].tolist()
                userId_list = unseen_args['UserID'].tolist()
                usa_node_list = list(distinct_syscalls_with_unseen_args)
                for i, key in enumerate(distinct_syscalls_with_unseen_args):
                    result['distinct_unseen_arg'] = usa_list[i]
                    result['usa node'] = usa_node_list[i]
                    result['user ID'] = userId_list[i]
                    result['uan_in_centrality'] = in_centrality[i]
                    result['uan_out_centrality'] = out_centrality[i]
                    result['Train uan_in_centrality'] = train_in_centrality[i]
                    result['Train uan_out_centrality'] = train_out_centrality[i]

                    result_array.append(result)
                    # tmp_result.append(result)
                # with open(os.path.join(os.path.join(save_dir, str(sequence_data.iloc[-1]['time']) + '_' + pname + '.json')),
                #           'w') as f:
                #     json.dump(tmp_result, f, indent=4, ensure_ascii=True, sort_keys=False)

    return uai

def only_sc_get_usi(df, train_seen_only_sc, scg, exploit_start_time, result):
    second_seen_only_sc = {}
    last_time = df.iloc[-1]['time']

    is_exploit = 0
    if exploit_start_time != 0:
        if last_time > exploit_start_time:
           is_exploit = 1

    for pname in df['ProcessName'].dropna().unique():
        if pname not in second_seen_only_sc:
            second_seen_only_sc[pname] = set()
        second_seen_only_sc[pname] = second_seen_only_sc[pname] | set(df[df.ProcessName == pname]['syscallInt'].unique())

    for pname in second_seen_only_sc:
        if pname not in train_seen_only_sc:
            continue
        unseen_sc_list = second_seen_only_sc[pname] - train_seen_only_sc[pname]

        for usc in unseen_sc_list:
            tmp_result = {}
            usc = str(usc)
            seq_u_in_centrality = nx.in_degree_centrality(scg._graphs[pname]).get(usc)
            seq_u_out_centrality = nx.out_degree_centrality(scg._graphs[pname]).get(usc)

            tmp_result['Process Name'] = pname
            tmp_result['exploit_start_time'] = exploit_start_time
            tmp_result['Current Time'] = last_time
            tmp_result['is_exploit'] = is_exploit

            tmp_result['in_centrality'] = seq_u_in_centrality
            tmp_result['out_centrality'] = seq_u_out_centrality

            result.append(tmp_result)

def get_usi(last_time, scg_second, scg_train, exploit_start_time, result, save_dir):
    usi = 0
    is_exploit = 0
    s_sc_node = set()
    t_sc_node = set()
    if exploit_start_time != 0:
        if last_time > exploit_start_time:
            is_exploit = 1

    for pname in scg_second._graphs:
        if pname not in scg_train._graphs:
            print(f'{pname} not in scg_train, Skip')
            continue
        # sc + ret or sc
        for u, v in scg_second._graphs[pname].edges():
            if not scg_train._graphs[pname].has_edge(u, v):
                tmp_result = {}
                s_sc_node.add(u)
                t_sc_node.add(v)
                has_u = scg_train._graphs[pname].has_node(str(u))
                has_v = scg_train._graphs[pname].has_node(str(v))

                seq_u_in_centrality = nx.in_degree_centrality(scg_second._graphs[pname]).get(u)
                seq_u_out_centrality = nx.out_degree_centrality(scg_second._graphs[pname]).get(u)

                seq_v_in_centrality = nx.in_degree_centrality(scg_second._graphs[pname]).get(v)
                seq_v_out_centrality = nx.out_degree_centrality(scg_second._graphs[pname]).get(v)

                train_u_in_centrality = nx.in_degree_centrality(scg_train._graphs[pname]).get(u)
                train_u_out_centrality = nx.out_degree_centrality(scg_train._graphs[pname]).get(u)

                train_v_in_centrality = nx.in_degree_centrality(scg_train._graphs[pname]).get(v)
                train_v_out_centrality = nx.out_degree_centrality(scg_train._graphs[pname]).get(v)

                tmp_result['Process Name'] = pname
                tmp_result['exploit_start_time'] = exploit_start_time
                tmp_result['Current Time'] = last_time
                tmp_result['is_exploit'] = is_exploit
                tmp_result['Node S'] = u
                tmp_result['S Exist'] = has_u
                tmp_result['Node T'] = v
                tmp_result['T Exist'] = has_v

                tmp_result['Seq S in'] = seq_u_in_centrality
                tmp_result['Seq S out'] = seq_u_out_centrality
                tmp_result['Seq T in'] = seq_v_in_centrality
                tmp_result['Seq T out'] = seq_v_out_centrality

                tmp_result['Train S in'] = train_u_in_centrality
                tmp_result['Train S out'] = train_u_out_centrality
                tmp_result['Train T in'] = train_v_in_centrality
                tmp_result['Train T out'] = train_v_out_centrality

                result.append(tmp_result)
                # if len(result) > 0:
                #     with open(os.path.join(os.path.join(save_dir, str(last_time) + '_' + pname + '.json')),
                #               'w') as f:
                #         json.dump(result, f, indent=4, ensure_ascii=True, sort_keys=False)

        usi += len(s_sc_node) + len(t_sc_node)

    return usi

def get_sc_size_by_df(df, exploit_start_time, sc_size_result):
    result = {}
    last_time = df.iloc[-1]['time']
    ret_result = [0] * len(SYSCALLS_ARGS_RET)

    is_exploit = 0
    if exploit_start_time != 0:
        if last_time > exploit_start_time:
            is_exploit = 1

    result['exploit_start_time'] = exploit_start_time
    result['Current Name'] = last_time
    result['is_exploit'] = is_exploit

    for i in range(len(SYSCALLS_ARGS_RET)):
        cur_seq_max = df[df.syscall == SYSCALLS_ARGS_RET[i]].RetReal.max()
        ret_result[i] = ret_result[i] if ret_result[i] > cur_seq_max else cur_seq_max
        result[SYSCALLS_ARGS_RET[i]] = ret_result[i]

    sc_size_result.append(result)
    return ret_result

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def initial_var(scenario_name, trainning_save_all_usi_usa, train_seen_args, train_scg, train_scg_only_sc, train_seen_only_sc, train_sc_size_list):
    # get Train Seen Args
    if len(train_seen_args) == 0:
        # if it is not training state, load from file, trainning state will save to the file
        if trainning_save_all_usi_usa is not True:
            print('Need load train_seen_args form File')
            with open(os.path.join(DATAOUT_DIR, scenario_name, 'All_Seen_Args.json')) as f:
                train_seen_args = json.load(f)

    # get Train Seen syscall graph
    if len(train_scg._graphs) == 0:
        # if it is not training state, load from file, trainning state will save to the file
        if trainning_save_all_usi_usa is not True:
            print('Need load train_scg form File')
            scg_dir = os.path.join(DATAOUT_DIR, scenario_name, 'SCG')
            # load all files in dir scg_dir
            for file in os.listdir(scg_dir):
                if file.endswith('.xml'):
                    pname = os.path.splitext(file)[0]
                    train_scg._graphs[pname] = nx.read_graphml(os.path.join(scg_dir, file))

            only_sc_scg_dir = os.path.join(scg_dir, 'ONLY_SC_SCG')
            for file in os.listdir(only_sc_scg_dir):
                if file.endswith('.xml'):
                    pname = os.path.splitext(file)[0]
                    train_scg_only_sc._graphs[pname] = nx.read_graphml(os.path.join(only_sc_scg_dir, file))

    # get Train Seen syscall， no return
    if len(train_seen_only_sc) == 0:
        if trainning_save_all_usi_usa is not True:
            print('Need load train_seen_only_sc From file')
            with open(os.path.join(os.path.join(DATAOUT_DIR, scenario_name, 'train_seen_sc_only_sc.json')), 'r') as f:
                train_seen_only_sc = json.load(f)
                for pname in train_seen_only_sc:
                    train_seen_only_sc[pname] = set(train_seen_only_sc[pname])

    # get Train syscall size maxsize
    if sum(train_sc_size_list) == 0:
        if trainning_save_all_usi_usa is not True:
            print('Need load train syscall size From file')
            with open(os.path.join(os.path.join(DATAOUT_DIR, scenario_name, 'train_sc_size.json')),
                      'r') as f:
                train_sc_size_list = json.load(f)

    return train_seen_args, train_scg, train_scg_only_sc, train_seen_only_sc, train_sc_size_list


def get_recording_expliot_time(dataloader, recording, data_type):
    exploit_start_time = 0
    recording_type = dataloader._metadata_list[data_type][recording.name]['recording_type']
    if recording_type == RecordingType.NORMAL_AND_ATTACK or recording_type == RecordingType.ATTACK:
        exploit_start_time = recording.metadata()["time"]["exploit"][0]["absolute"] * (10 ** 9)
        if exploit_start_time == 0:
            print('!!!!!!!!!!!!! exploit_start_time! why it is 0')

    return exploit_start_time

def check_work_dir(training, recoding_path):

    data_path = os.path.join(recoding_path, 'FinalData')
    dataframe_path = os.path.join(recoding_path, 'DataFrame')
    scg_path = os.path.join(recoding_path, 'SCG')

    if not os.path.exists(recoding_path):
        os.makedirs(recoding_path)
        os.mkdir(dataframe_path)
        os.mkdir(scg_path)
        os.mkdir(data_path)

    seenarg_path = seensc_path = scsize_path = None

    if training is not True:
        seenarg_path = os.path.join(recoding_path, 'SA')
        if not os.path.exists(seenarg_path):
            os.mkdir(seenarg_path)

        seensc_path = os.path.join(recoding_path, 'SSC')
        if not os.path.exists(seensc_path):
            os.mkdir(seensc_path)

        scsize_path = os.path.join(recoding_path, 'SSIZE')
        if not os.path.exists(scsize_path):
            os.mkdir(scsize_path)

    return data_path, dataframe_path, scg_path, seenarg_path, seensc_path, scsize_path

def get_training_data_feature(sc_df, train_sc_size_list, train_scg, train_scg_only_sc, train_seen_only_sc, train_seen_args):
    ret = get_sc_size_by_df(sc_df, 0, [])
    train_sc_size_list = np.fmax(train_sc_size_list, ret)
    # get syscall graph use syscall + ret
    sc_df.apply(df_create_scgraph, args=(train_scg,), axis=1)
    # get syscall graph use syscall only
    sc_df.apply(df_create_scgraph, args=(train_scg_only_sc, True,), axis=1)
    # get syscall seened in training data
    for pname in sc_df['ProcessName'].dropna().unique():
        tmp_df = sc_df[sc_df.ProcessName == pname]
        if pname not in train_seen_only_sc:
            train_seen_only_sc[pname] = set()
        train_seen_only_sc[pname] = train_seen_only_sc[pname] | set(tmp_df['syscallInt'].unique())

        if pname not in train_seen_args:
            train_seen_args[pname] = set()
        train_seen_args[pname] = train_seen_args[pname] | set(tmp_df['Params'].dropna().unique())

    return train_sc_size_list, train_scg, train_scg_only_sc, train_seen_only_sc, train_seen_args

def save_second_data(recoding_path, numpy_all, df_all, seenarg_path, usa_result, seensc_path, usc_result, only_sc_usc_edges_result, only_sc_usc_result, scsize_path, sc_size_result):
    if len(numpy_all) > 0:
        np.save(os.path.join(recoding_path, 'array.npy'), np.array(numpy_all, dtype=object))

        if len(df_all) != len(numpy_all):
            print('!!! df_all len is not equal numpy_ALL')

        # save df_all
        save_dict(df_all, os.path.join(recoding_path, 'df_all'))

    # if usa_result is not empty, save it to file in SA dir for per recoding
    if len(usa_result) > 0:
        with open(os.path.join(seenarg_path, 'USA.json'), 'w') as f:
            json.dump(usa_result, f, indent=4, ensure_ascii=True, sort_keys=False, default=default_dump)

    # if usc_result is not empty, save it to file in SSC dir for per recoding
    if len(usc_result) > 0:
        with open(os.path.join(seensc_path, 'USC.json'), 'w') as f:
            json.dump(usc_result, f, indent=4, ensure_ascii=True, sort_keys=False, default=default_dump)

    # if only_sc_usc_edges_result is not empty, save it to file in SSC dir for per recoding
    if len(only_sc_usc_edges_result) > 0:
        with open(os.path.join(seensc_path, 'ONLY_SC_USC.json'), 'w') as f:
            json.dump(only_sc_usc_edges_result, f, indent=4, ensure_ascii=True, sort_keys=False, default=default_dump)

    # if only_sc_usc_result is not empty, save it to file in SSC dir for per recoding
    if len(only_sc_usc_result) > 0:
        with open(os.path.join(seensc_path, 'OSC_USC.json'), 'w') as f:
            json.dump(only_sc_usc_result, f, indent=4, ensure_ascii=True, sort_keys=False, default=default_dump)

    # if sc_size_result is not empty, save it to file in SSIZE dir for per recoding
    if len(sc_size_result) > 0:
        with open(os.path.join(scsize_path, 'SC_SIZE.json'), 'w') as f:
            json.dump(sc_size_result, f, indent=4, ensure_ascii=True, sort_keys=False, default=default_dump)

# Trining result file path:
# DATAOUT_DIR/scenario_name/All_Seen_Args.json
# DATAOUT_DIR/scenario_name/SCG/pname.xml
# DATAOUT_DIR/scenario_name/ONLY_SC_SCG/pname.xml
# DATAOUT_DIR/scenario_name/train_seen_sc_only_sc.json
# DATAOUT_DIR/scenario_name/SSIZE/train_sc_size.json

def save_training_data(training, train_sc_size_list, train_scg, train_scg_only_sc, train_seen_only_sc, train_seen_args, scenario_name):

    if not training:
        return

    # if it is in trainning state, we need to save train_seen_args to file
    if len(train_seen_args) == 0:
        print('Training... train_seen_args is empty, skip !')
    else:
        for pname in train_seen_args:
            train_seen_args[pname] = list(train_seen_args[pname])

        with open(os.path.join(os.path.join(DATAOUT_DIR, scenario_name, 'All_Seen_Args.json')), 'w') as f:
            json.dump(train_seen_args, f, indent=4, ensure_ascii=True, sort_keys=False)

    # if it is in trainning state, we need to save train_scg to file
    if len(train_scg._graphs) == 0:
        print('Training... train_scg is empty, skip !')
    else:
        scg_dir = os.path.join(DATAOUT_DIR, scenario_name, 'SCG')
        for pname in train_scg._graphs:
            if not os.path.exists(scg_dir):
                os.mkdir(scg_dir)
            nx.write_graphml(train_scg._graphs[pname], os.path.join(scg_dir, pname + '.xml'), encoding='utf-8')

        only_sc_scg_dir = os.path.join(scg_dir, 'ONLY_SC_SCG')
        if not os.path.exists(only_sc_scg_dir):
            os.mkdir(only_sc_scg_dir)

        for pname in train_scg_only_sc._graphs:
            nx.write_graphml(train_scg_only_sc._graphs[pname], os.path.join(only_sc_scg_dir, pname + '.xml'),
                            encoding='utf-8')

    # if it is in trainning state, we need to save train_seen_only_sc to file
    if len(train_seen_only_sc) == 0:
        print('Training... train_seen_only_sc is empty, skip!')
    else:
        if np.nan in train_seen_only_sc:
            train_seen_only_sc['None'] = train_seen_only_sc.pop(np.nan)

        tmp_dict = {}
        for pname in train_seen_only_sc:
            tmp_dict[pname] = list(train_seen_only_sc[pname])

        with open(os.path.join(os.path.join(DATAOUT_DIR, scenario_name, 'train_seen_sc_only_sc.json')),
                  'w') as f:
            json.dump(tmp_dict, f, indent=4, ensure_ascii=True, sort_keys=False)

    # if it is in trainning state, we need to save train_seen_only_sc_usc to file
    if sum(train_sc_size_list) > 0:
        train_sc_size_list = train_sc_size_list.tolist()
        with open(os.path.join(os.path.join(DATAOUT_DIR, scenario_name, 'train_sc_size.json')),
                  'w') as f:
            json.dump(train_sc_size_list, f, indent=4, ensure_ascii=True, sort_keys=False)

# def main_func(scenarios_pos_start:int, scenarios_pos_end:int):
if __name__ == '__main__':

    # scenarios ordered by training data size asc
    SCENARIOS = [
        "Bruteforce_CWE-307",
        "CWE-89-SQL-injection",
        "CVE-2012-2122",
        "CVE-2017-7529",
        "PHP_CWE-434",
        "EPS_CWE-434",
        "CVE-2014-0160",

        "CVE-2018-3760",
        "CVE-2019-5418",

        # End
        "CVE-2020-23839",
        "ZipSlip",
        "CVE-2020-9484",
        "Juice-Shop",
        "CVE-2020-13942",
        "CVE-2017-12635_6"
    ]
    # SCENARIO_RANGE = SCENARIOS[scenarios_pos_start:scenarios_pos_end]
    SCENARIO_RANGE = SCENARIOS[0:1]

    for scenario_name in SCENARIO_RANGE:
        print('Handle scenario %s' %scenario_name)
        scenario_path = os.path.join(work_dir, scenario_name)
        # 如果 GP_DATA_DIR 目录下不存在 scenario_name 目录
        # if GP_DATA_DIR path do not exist scenario_name dir
        if not os.path.exists(os.path.join(DATAOUT_DIR, scenario_name)):
            print('scenario_path not exist, Create')
            os.mkdir(os.path.join(DATAOUT_DIR, scenario_name))

        # 保证训练完毕
        intEmbed = IntEmbedding(scenario_name=scenario_name)
        pnr = ProcessNameAndRet(intEmbed)
        dataloader = dataloader_factory(scenario_path, direction=Direction.CLOSE)
        # train or load intembedding model
        Init_intEmbed(dataloader, scenario_name, intEmbed)

        # features
        # 'time','UserID', 'PID', 'ProcessName', 'TID', 'syscall', 'DIR', 'ARGS'
        # 1. 每秒 时间间隔 数量， 针对每一项进行频率统计
        # 2. PID and TID switch frequency
        # 3. syscall call graph create, user intEmbedding + return value(default 0)
        # TODO: ARGs Analysis
        ##################
        data_type_list = [
            {'type' : TRAINING, 'data' : dataloader.training_data()},
            {'type' : VALIDATION, 'data' :dataloader.validation_data()},
            {'type' : TEST, 'data' :dataloader.test_data()}
        ]
        # save training data seen args, some syscall files
        train_seen_args = {}
        # save training data seen syscall, just only syscall int
        train_seen_only_sc = {}
        # save training data seen syscall graph
        train_scg = m_SystemCallGraph()
        # save training data seen syscall graph, just only syscall int
        train_scg_only_sc = m_SystemCallGraph()
        # save training data , some syscall return size
        train_sc_size_list = [0] * len(SYSCALLS_ARGS_RET)

        for dict in data_type_list:
            print('Handle file type %s' % dict['type'])
            # indicate it is trainning data when it is true
            training = False
            if dict['type'] == TRAINING:
                training = True
                print('Trainning save all usi usa')

            # 初始化变量
            train_seen_args, train_scg, train_scg_only_sc, train_seen_only_sc, train_sc_size_list = initial_var(scenario_name, training, train_seen_args, train_scg, train_scg_only_sc, train_seen_only_sc, train_sc_size_list)

            for recording in tqdm(dict['data'],
                                  f"Handle {dict['type']} DataSet".rjust(27),
                                  unit=" recording_train"):



                pathlist = recording.path.split('\\')
                if dict['type'] != TEST:
                    recoding_path = os.path.join(DATAOUT_DIR, scenario_name, pathlist[-2], recording.name)
                else:
                    # for test , test dir has two subdir
                    recoding_path = os.path.join(DATAOUT_DIR, scenario_name, pathlist[-3], pathlist[-2], recording.name)

                if os.path.exists(recoding_path):
                    print('Skip recoding_path %s' % recoding_path)
                    continue

                exploit_start_time = get_recording_expliot_time(dataloader, recording, dict['type'])
                data_path, dataframe_path, scg_path, seenarg_path, seensc_path, scsize_path = check_work_dir(training, recoding_path)

                # names = ['time','UserID', 'PID', 'ProcessName', 'TID', 'syscall', 'DIR', 'RET', 'ARGS', 'syscallInt', 'Ret', 'RetReal', 'Params'])
                for sc_df in recording.syscalls_df():
                    # 对于每一个文件
                    index = 0
                    # save all numpy data
                    numpy_all = []
                    # save all dataframe
                    df_all = {}
                    # save all un seen args
                    usa_result = []
                    # save all syscall and ret
                    usc_result = []
                    # save all un seen syscall edges
                    only_sc_usc_edges_result = []
                    # save all un seen syscall
                    only_sc_usc_result = []
                    sc_size_result = []

                    if len(sc_df) < 30:
                        print('Too Short %d' % len(sc_df))
                        continue
                    # just handle close direction
                    sc_df = sc_df[sc_df['DIR'] == '<'].copy()
                    # get four data once
                    sc_df[['syscallInt', 'Ret', 'RetReal', 'Params']] = sc_df.apply(df_get_other_feature, args=(pnr,), axis=1, result_type='expand')

                    start_time = sc_df.iloc[0]['time']
                    end_time = start_time + 1000000000

                    if training:
                        train_sc_size_list, train_scg, train_scg_only_sc, train_seen_only_sc, train_seen_args = get_training_data_feature(sc_df, train_sc_size_list, train_scg, train_scg_only_sc, train_seen_only_sc, train_seen_args)

                    while sc_df.iloc[-1]['time'] > end_time:
                        # handle 1 S period data
                        df = sc_df[(sc_df['time'] >= start_time) & (sc_df['time'] < end_time)]
                        # 如果序列长度小于 30， 跳过
                        if len(df) < 30:
                            start_time += 200000000
                            end_time = start_time + 1000000000
                            continue

                        last_time = str(df.iloc[-1]['time'])
                        data = [index, last_time]
                        second_seen_args = {}
                        # save dataframe to file
                        # df.to_pickle(os.path.join(dataframe_path, str(df.iloc[-1]['time']) + '.pkl'))
                        df_all[index] = df
                        # 39 bytes Time Delta
                        data.extend(cal_time_delte(df))
                        # 4 bytes PID TID Feature
                        data.extend(cal_pid_tid_switch_frequency(df))

                        if not training:
                            # if it is not trainning

                            # SC SIZE
                            # get syscall size for each second
                            get_sc_size_by_df(df, exploit_start_time, sc_size_result)

                            # SCG
                            # we need to get scg for each second
                            scg_second = create_scg_for_second(df, scg_path)

                            only_sc_scg = m_SystemCallGraph()
                            df.apply(df_create_scgraph, args=(only_sc_scg, True), axis=1)

                            # if it is not trainning state, we need to get seen args for each process
                            uai = get_uai(df, train_seen_args, only_sc_scg, train_scg_only_sc,
                                                   exploit_start_time, usa_result, seenarg_path)

                            # USI
                            get_usi(df.iloc[-1]['time'], scg_second, train_scg, exploit_start_time, usc_result,
                                    seensc_path)
                            # only syscall
                            usi = get_usi(df.iloc[-1]['time'], only_sc_scg, train_scg_only_sc, exploit_start_time,
                                    only_sc_usc_edges_result,
                                    seensc_path)
                            # 只是对未见的系统调用进行统计，不包括未见的边
                            only_sc_get_usi(df, train_seen_only_sc, only_sc_scg, exploit_start_time, only_sc_usc_result)
                        else:
                            # append UAI for trainning data
                            uai = 0
                            usi = 0

                        data.append(uai)
                        data.append(usi)
                        # ret number
                        data.append(len(df[df.Ret < 0]) / len(df))
                        # syscall freq max 8
                        sc_freq_max = df.syscallInt.value_counts().index.tolist()[:8]
                        data.extend(sc_freq_max)
                        # data 0 index
                        # data 1 time
                        # data [2:41] timedelta
                        # data [41:45] pid/tid
                        # data [45:47] uai usi
                        # data [47:48] return less 0 number
                        # data [48:56] 8 bytes syscall max freq
                        # index_start = 0
                        # index_time  = 1
                        # index_timedelta_start = 2
                        # index_timdelta_end = 41
                        # index_ptid_start = 41
                        # index_ptid_end = 45
                        # index_uai = 45
                        # index_usi = 46
                        # index_ret_total = 47
                        # index_sc_max_start = 48
                        # index_sc_max_end = 56

                        numpy_all.append(data)
                        index += 1

                        start_time += 200000000
                        end_time = start_time + 1000000000

                save_second_data(recoding_path, numpy_all, df_all, seenarg_path, usa_result, seensc_path, usc_result, only_sc_usc_edges_result, only_sc_usc_result, scsize_path, sc_size_result)

            save_training_data(training, train_sc_size_list, train_scg, train_scg_only_sc, train_seen_only_sc, train_seen_args, scenario_name)
    # scenario END
    print('End')
#
# if __name__ == '__main__':
#     process_list = []
#
#     for i in range(7):
#         p = Process(target=main_func, args=(i, i+1))
#         p.start()
#         process_list.append(p)
#
#     for i in process_list:
#         p.join()
#
#     print('ALL Create DataSet exit')