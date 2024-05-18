
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from rich.table import Table
from rich.console import Console
import scienceplots

from dataloader.direction import Direction
from dataloader.dataloader_factory import dataloader_factory
from dataloader.data_loader_2021 import DataLoader2021
from dataloader.data_loader_2021 import RecordingType
from dataloader.dataset_create_gp import DATAOUT_DIR, work_dir
from PIL import Image
import io
from matplotlib.font_manager import FontProperties

# plt.style.use(['science','ieee'])
plt.style.use(['science','grid'])
plt.style.use(['science','no-latex'])
plt.rcParams['axes.unicode_minus'] = False
# 修改图中的默认字体
# plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimSun']
font_set = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')

NORMAL = 'NORMAL'
NORMAL_AND_ATTACK = 'NORMAL_AND_ATTACK'
ATTACK = 'ATTACK'
IDLE = 'IDLE'

TRAINING = 'training'
VALIDATION = 'validation'
TEST = 'test'

def create_tiff(save_path):
    png1 = io.BytesIO()
    plt.savefig(png1, format='png', dpi=100)
    png2 = Image.open(png1)
    # Save as TIFF
    png2.save(save_path)
    png1.close()
    plt.clf()

def display_file(train_file_dict, val_file_dict, test_file_dict, scenario_name):
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

    console = Console(record=True)
    console.print(output_table, justify="center")
    console.save_svg(os.path.join(DATAOUT_DIR, scenario_name, 'File_Statistics.svg'), title=scenario_name)
# 统计文件个数与类型
def get_file_num(dataloader, scenario_name):
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

    display_file(train_file_dict, val_file_dict, test_file_dict, scenario_name)
def handle_unsc_group_by_time_in_out_degree(group_df):
    uss = len(set(group_df['Node S']))
    ust = len(set(group_df['Node T']))

    return (uss + ust) * (group_df[['Train S in', 'Train T in', 'Train S out', 'Train T out']].sum().sum())
    # return (uss * group_df[['Train S in', 'Train S out']].sum().sum()) + (ust * group_df[['Train T in', 'Train T out']].sum().sum())

def handle_unsc_group_by_time_other(group_df):
    usi = group_df['usi degree']
    # return (uss + ust) * (group_df[['Train S in', 'Train T in', 'Train S out', 'Train T out']].sum().sum())
    # return (uss * group_df[['Train S in', 'Train S out']].sum().sum()) + (ust * group_df[['Train T in', 'Train T out']].sum().sum())
    return usi

def handle_unsc_group_by_time(group_df):
    # uss = len(set(group_df['Node S']))
    # ust = len(set(group_df['Node T']))
    usi = group_df['usi']
    # return (uss + ust) * (group_df[['Train S in', 'Train T in', 'Train S out', 'Train T out']].sum().sum())
    # return (uss * group_df[['Train S in', 'Train S out']].sum().sum()) + (ust * group_df[['Train T in', 'Train T out']].sum().sum())
    return usi

def get_uai_per_row(group_df):
    uai = group_df['uai'].max()
    # / len(set(group_df['Procee Name']))
    num_uai_max = group_df['sc unseen args num'].product()
    return uai

def analyze_dir_df(df, recoding_path, type_name):

    unseenType_path = os.path.join(recoding_path, type_name)
    # list json  filr in unseenSC_path
    if os.path.exists(unseenType_path) == False:
        return df
    # Now, just USC.json or USA.json one file
    json_list = os.listdir(unseenType_path)
    if len(json_list) > 0:
        for jsfile in json_list:
            if jsfile.endswith('.json'):
                json_path = os.path.join(unseenType_path, jsfile)
                df = df.append(pd.read_json(json_path))
    return df

def analyze_onefile_by_dir_df(df, recoding_path, type_name, file_name):
    json_path = os.path.join(recoding_path, type_name, file_name)
    # list json  filr in unseenSC_path
    if os.path.exists(json_path) == False:
        return df
    # Now, just USC.json or USA.json one file
    df = df.append(pd.read_json(json_path))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    return df

def analysis_unseen_args(dataloader, scenario_name):
    data_type_list = [
        {'type': TRAINING, 'data': dataloader.training_data()},
        {'type': VALIDATION, 'data': dataloader.validation_data()},
        {'type': TEST, 'data': dataloader.test_data()}
    ]

    unseenargs_df =  pd.DataFrame()

    for data_type in data_type_list:
        for recording in tqdm(data_type['data'],
                              f"Load UNSEEN ARGS {data_type['type']}".rjust(27)):
            pathlist = recording.path.split('\\')
            if data_type['type'] != TEST:
                recoding_path = os.path.join(DATAOUT_DIR, scenario_name, pathlist[-2], recording.name)
            else:
                # for test , test dir has two subdir
                recoding_path = os.path.join(DATAOUT_DIR, scenario_name, pathlist[-3], pathlist[-2], recording.name)

            if data_type['type'] != TRAINING:
                unseenargs_df = analyze_onefile_by_dir_df(unseenargs_df, recoding_path, 'SA', 'USA.json')

    # save seensc_df to csv
    if len(unseenargs_df) > 0:
        # unseenargs_df['Current Time'] = pd.to_datetime(unseenargs_df['Current Time'])
        unseenargs_df.to_csv(os.path.join(DATAOUT_DIR, scenario_name, 'unseenArgs.csv'))

        # nor_result = unseenargs_df[unseenargs_df.is_exploit == False]['uai'].value_counts()
        # exp_result = unseenargs_df[unseenargs_df.is_exploit == True]['uai'].value_counts()
        nor_result = unseenargs_df[unseenargs_df.is_exploit == False].groupby('Current Time').apply(get_uai_per_row)
        exp_result = unseenargs_df[unseenargs_df.is_exploit == True].groupby('Current Time').apply(get_uai_per_row)

        # plt.figure(figsize=(18, 15))
        # plt.title("UnSeen ARGS Score")
        plt.plot(nor_result.value_counts().index.tolist(), nor_result.value_counts().values.tolist(), c='b', alpha=0.5, marker='o', ls='', label='正常序列')
        plt.plot(exp_result.value_counts().index.tolist(), exp_result.value_counts().values.tolist(), c='r', alpha=0.5, marker='v',  ls='', label='异常序列')
        plt.xlabel("未见文件异常分数", fontproperties=font_set)
        plt.ylabel("周期样本数量", fontproperties=font_set)
        plt.legend(prop=font_set)
        create_tiff(os.path.join(DATAOUT_DIR, scenario_name, 'UnSeenArgs.tiff'))
    print('analysis_unseen_args() Success')

def analysis_syscall_size(dataloader, scenario_name):
    data_type_list = [
        {'type': TRAINING, 'data': dataloader.training_data()},
        {'type': VALIDATION, 'data': dataloader.validation_data()},
        {'type': TEST, 'data': dataloader.test_data()}
    ]

    sc_size_df = pd.DataFrame()

    for data_type in data_type_list:
        for recording in tqdm(data_type['data'],
                              f"Load SYSCALL SIZE {data_type['type']}".rjust(27)):
            pathlist = recording.path.split('\\')
            if data_type['type'] != TEST:
                recoding_path = os.path.join(DATAOUT_DIR, scenario_name, pathlist[-2], recording.name)
            else:
                # for test , test dir has two subdir
                recoding_path = os.path.join(DATAOUT_DIR, scenario_name, pathlist[-3], pathlist[-2], recording.name)

            # if data_type['type'] != TRAINING:
            sc_size_df = analyze_onefile_by_dir_df(sc_size_df, recoding_path, 'SSIZE', 'SC_SIZE.json')
    # save seensc_df to csv
    # sc_size_df['Current Name'] = pd.to_datetime(unseenargs_df['Current Name'])
    sc_size_df.to_csv(os.path.join(DATAOUT_DIR, scenario_name, 'sysCallSize.csv'))
    sc_size_df[sc_size_df.is_exploit == 0].describe().to_csv(
        os.path.join(DATAOUT_DIR, scenario_name, 'sysCallSize_normal.csv'))
    sc_size_df[sc_size_df.is_exploit == 1].describe().to_csv(
        os.path.join(DATAOUT_DIR, scenario_name, 'sysCallSize_exploit.csv'))
    print('analysis_syscall_size() Success')

def analysis_ret_max(dataloader, scenario_name):
    data_type_list = [
        # {'type': TRAINING, 'data': dataloader.training_data()},
        {'type': VALIDATION, 'data': dataloader.validation_data()},
        {'type': TEST, 'data': dataloader.test_data()}
    ]

    for data_type in data_type_list:
        freq_nor_list = []
        freq_exp_list = []
        for recording in tqdm(data_type['data'],
                              f"RetNumber Load DataSet Array {data_type['type']}".rjust(27)):

            exploit_start_time = 0
            recording_type = dataloader._metadata_list[data_type['type']][recording.name]['recording_type']
            if recording_type == RecordingType.NORMAL_AND_ATTACK or recording_type == RecordingType.ATTACK:
                exploit_start_time = recording.metadata()["time"]["exploit"][0]["absolute"] * (10 ** 9)
                if exploit_start_time == 0:
                    print('!!!!!!!!!!!!! exploit_start_time! why it is 0')

            for (data_array, df) in recording.df_and_np():
                if data_array is None:
                    continue

                for data in data_array:
                    # ret = data[47]
                    index = data[0]
                    freq = len(df[index])
                    if exploit_start_time == 0 or exploit_start_time > df[index].iloc[-1]['time']:
                    #     # plt.scatter(x_labels, y, c='b', alpha=0.5, s=6)
                    #     plt.plot(x_labels, y, c='b', alpha=0.5, marker='o', markersize=6, ls='')
                        freq_nor_list.append(freq)
                    else:
                        freq_exp_list.append(freq)
                    #     plt.plot(x_labels, y, c='r', alpha=0.5, marker='v', markersize=9, ls='')
                    #     # plt.scatter(x_labels, y, c='r', alpha=0.5, s=10)

    # print('analysis_ret_max() Success')

    freq_nor_list = np.array(freq_nor_list)
    freq_exp_list = np.array(freq_exp_list)
    y_list = np.bincount(freq_nor_list)
    x_labels = [x for x in range(len(y_list))]
    # plt.figure(figsize=(18, 15))
    # plt.title("Return Status")
    plt.plot(x_labels , y_list, c='b', alpha=0.5, marker='^', label='正常序列')
    y_list = np.bincount(freq_exp_list)
    x_labels = [x for x in range(len(y_list))]
    plt.plot(x_labels, y_list, c='r', alpha=0.2, marker='o',label='异常序列')

    plt.xlabel("频率大小", fontproperties=font_set)
    plt.ylabel("序列个数", fontproperties=font_set)
    plt.legend(prop=font_set)

    create_tiff(os.path.join(DATAOUT_DIR, scenario_name, 'Syscall_Freq.tiff'))
    # plt.savefig(os.path.join(DATAOUT_DIR, scenario_name, 'Return.png'))
    # print('Return Status Success')
    print('syscall Frequency')

def analysis_sc_max_freq(dataloader, scenario_name):
    data_type_list = [
        {'type': TRAINING, 'data': dataloader.training_data()},
        {'type': VALIDATION, 'data': dataloader.validation_data()},
        {'type': TEST, 'data': dataloader.test_data()}
    ]

    # plt.figure(figsize=(18, 15))
    plt.title("SC Max Call Freq")
    x_labels = [i for i in range(8)]
    dataSet_Static = {TRAINING:0, VALIDATION:0, TEST:0}
    for data_type in data_type_list:
        for recording in tqdm(data_type['data'],
                              f"ScMaxFreq Load DataSet Array {data_type['type']}".rjust(27)):

            exploit_start_time = 0
            recording_type = dataloader._metadata_list[data_type['type']][recording.name]['recording_type']
            if recording_type == RecordingType.NORMAL_AND_ATTACK or recording_type == RecordingType.ATTACK:
                exploit_start_time = recording.metadata()["time"]["exploit"][0]["absolute"] * (10 ** 9)
                if exploit_start_time == 0:
                    print('!!!!!!!!!!!!! exploit_start_time! why it is 0')

            for (data_array, df) in recording.df_and_np():
                if data_array is None:
                    continue

                dataSet_Static[data_type['type']] += len(data_array)

                for data in data_array:
                    index = data[0]
                    y = data[48:56]
                    if len(y) != 8:
                        print('!!!!!!!!!!!!! len(y) %d != 8' % len(y))
                        continue

                    if exploit_start_time == 0 or exploit_start_time > df[index].iloc[-1]['time']:
                        # plt.scatter(x_labels, y, c='b', alpha=0.5, s=6)
                        plt.plot(x_labels, y, c='b', alpha=0.5, marker='o', ls='', label='正常序列')
                    else:
                        plt.plot(x_labels, y, c='r', alpha=0.5, marker='v', ls='', label='异常序列')
                        # plt.scatter(x_labels, y, c='r', alpha=0.5, s=10)

    plt.xlabel("频率序列", fontproperties=font_set)
    plt.ylabel("频率", fontproperties=font_set)
    plt.legend(prop=font_set)
    # create_tiff(os.path.join(DATAOUT_DIR, scenario_name, 'SC_Max_Call_Freq.tiff'))
    # plt.show()
    print('analysis_sc_max_freq() Success')

    output_table = Table(title='DataSet Statistics')
    output_table.add_column("", style="magenta")
    output_table.add_column("Training", style="magenta")
    output_table.add_column("Val", style="magenta")
    output_table.add_column("Test Normal", style="magenta")
    output_table.add_column("Test Exploit", style="magenta")
    output_table.add_row('Total', str(dataSet_Static[TRAINING]), str(dataSet_Static[VALIDATION]), str(dataSet_Static[TEST]), str(dataSet_Static[TEST]))

    console = Console(record=True)
    console.print(output_table, justify="center")
    console.save_svg(os.path.join(DATAOUT_DIR, scenario_name + '_DataSet_Statistics.svg'), title=scenario_name)

    print('Dataset Statistics Success')

# get unseen SC and Dataset Statistics
def analysis_unseen_sc(dataloader, scenario_name):
    data_type_list = [
        {'type': TRAINING, 'data': dataloader.training_data()},
        {'type': VALIDATION, 'data': dataloader.validation_data()},
        {'type': TEST, 'data': dataloader.test_data()}
    ]

    # unseensc_df = pd.DataFrame(columns=['exploit_time', 'is_exploit', 'time', 'pname', 'src', 'tar', 'src exits', 'tar exits', 'src degree', 'tar degree'])
    unseensc_df = pd.DataFrame()
    unseensc_only_sc_df = pd.DataFrame()
    for data_type in data_type_list:
        for recording in tqdm(data_type['data'],
                              f"Load UNSEEN SC {data_type['type']}".rjust(27)):
            pathlist = recording.path.split('\\')
            if data_type['type'] != TEST:
                recoding_path = os.path.join(DATAOUT_DIR, scenario_name, pathlist[-2], recording.name)
            else:
                # for test , test dir has two subdir
                recoding_path = os.path.join(DATAOUT_DIR, scenario_name, pathlist[-3], pathlist[-2], recording.name)

            if data_type['type'] != TRAINING:
                unseensc_df = analyze_onefile_by_dir_df(unseensc_df, recoding_path, 'SSC', 'USC.json')
                unseensc_only_sc_df = analyze_onefile_by_dir_df(unseensc_only_sc_df, recoding_path, 'SSC', 'ONLY_SC_USC.json')


    # save seensc_df to csv
    # unseensc_df['Current Time'] = pd.to_datetime(unseensc_df['Current Time'])
    # unseensc_only_sc_df['Current Time'] = pd.to_datetime(unseensc_only_sc_df['Current Time'])
    try:
        unseensc_df.to_csv(os.path.join(DATAOUT_DIR, scenario_name, 'unseenSc.csv'))
        unseensc_only_sc_df.to_csv(os.path.join(DATAOUT_DIR, scenario_name, 'unseenSc_only_sc.csv'))
    except:
        print('Save file ERROR')

    # nor_result = unseensc_df[unseensc_df.is_exploit == 0].groupby('Current Time').apply(handle_unsc_group_by_time)
    # exp_result = unseensc_df[unseensc_df.is_exploit == 1].groupby('Current Time').apply(handle_unsc_group_by_time)
    # nor_result_only_sc = unseensc_only_sc_df[unseensc_only_sc_df.is_exploit == 0].groupby('Current Time').apply(handle_unsc_group_by_time)
    # exp_result_only_sc = unseensc_only_sc_df[unseensc_only_sc_df.is_exploit == 1].groupby('Current Time').apply(handle_unsc_group_by_time)
    nor_result = unseensc_df[unseensc_df.is_exploit == 0].apply(handle_unsc_group_by_time_other, axis=1)
    exp_result = unseensc_df[unseensc_df.is_exploit == 1].apply(handle_unsc_group_by_time_other, axis=1)
    nor_result_only_sc = unseensc_only_sc_df[unseensc_only_sc_df.is_exploit == 0].apply(handle_unsc_group_by_time_other, axis=1)
    exp_result_only_sc = unseensc_only_sc_df[unseensc_only_sc_df.is_exploit == 1].apply(handle_unsc_group_by_time_other, axis=1)

    us_sc_ret_nor_max = nor_result.max()
    us_sc_ret_exp_min = exp_result.min()
    us_sc_nor_max = nor_result_only_sc.max()
    us_sc_exp_min = exp_result_only_sc.min()

    output_table = Table(title='UnSeen Syscall Statistics')
    output_table.add_column("TYPE", style="magenta")
    output_table.add_column("NORMAL MAX", style="magenta")
    output_table.add_column("EXPLOIT MIN", style="magenta")
    output_table.add_column("RESULT", style="magenta")
    output_table.add_column("Percentage", style="magenta")

    output_table.add_row('Syscall & Ret', str(us_sc_ret_nor_max), str(us_sc_ret_exp_min), 'OK' if us_sc_ret_exp_min > us_sc_ret_nor_max else 'FAIL', str(len(exp_result[exp_result > us_sc_ret_nor_max]) / len(exp_result)))
    output_table.add_row('Syscall', str(us_sc_nor_max), str(us_sc_exp_min), 'OK' if us_sc_exp_min > us_sc_nor_max else 'FAIL', str(len(exp_result_only_sc[exp_result_only_sc > us_sc_nor_max]) / len(exp_result_only_sc)))

    console = Console(record=True)
    console.print(output_table, justify="center")
    console.save_svg(os.path.join(DATAOUT_DIR, scenario_name, 'UnSeenSyc_Statistics_other.svg'), title=scenario_name)


    nor_result = unseensc_df[unseensc_df.is_exploit == 0].apply(handle_unsc_group_by_time, axis=1)
    exp_result = unseensc_df[unseensc_df.is_exploit == 1].apply(handle_unsc_group_by_time, axis=1)
    nor_result_only_sc = unseensc_only_sc_df[unseensc_only_sc_df.is_exploit == 0].apply(handle_unsc_group_by_time,
                                                                                        axis=1)
    exp_result_only_sc = unseensc_only_sc_df[unseensc_only_sc_df.is_exploit == 1].apply(handle_unsc_group_by_time,
                                                                                        axis=1)

    us_sc_ret_nor_max = nor_result.max()
    us_sc_ret_exp_min = exp_result.min()
    us_sc_nor_max = nor_result_only_sc.max()
    us_sc_exp_min = exp_result_only_sc.min()

    output_table = Table(title='UnSeen Syscall Statistics')
    output_table.add_column("TYPE", style="magenta")
    output_table.add_column("NORMAL MAX", style="magenta")
    output_table.add_column("EXPLOIT MIN", style="magenta")
    output_table.add_column("RESULT", style="magenta")
    output_table.add_column("Percentage", style="magenta")

    output_table.add_row('Syscall & Ret', str(us_sc_ret_nor_max), str(us_sc_ret_exp_min),
                         'OK' if us_sc_ret_exp_min > us_sc_ret_nor_max else 'FAIL',
                         str(len(exp_result[exp_result > us_sc_ret_nor_max]) / len(exp_result)))
    output_table.add_row('Syscall', str(us_sc_nor_max), str(us_sc_exp_min),
                         'OK' if us_sc_exp_min > us_sc_nor_max else 'FAIL',
                         str(len(exp_result_only_sc[exp_result_only_sc > us_sc_nor_max]) / len(exp_result_only_sc)))

    console = Console(record=True)
    console.print(output_table, justify="center")
    console.save_svg(os.path.join(DATAOUT_DIR, scenario_name, 'UnSeenSyc_Statistics.svg'), title=scenario_name)

    # plt.figure(figsize=(18, 15))
    # plt.title("UnSeen Score")
    plt.plot(nor_result.value_counts().index.tolist(), nor_result.value_counts().values.tolist(), c='b', alpha=0.5, marker='o',  ls='', label='正常序列')
    plt.plot(exp_result.value_counts().index.tolist(), exp_result.value_counts().values.tolist(), c='r', alpha=0.5, marker='v',  ls='', label='异常序列')
    plt.xlabel('未见系统调用异常分数', fontproperties=font_set)
    plt.ylabel('周期样本数量', fontproperties=font_set)
    plt.legend(prop=font_set)
    create_tiff('UnSeenSc.tiff')
    # plt.title("UnSeen Only SC Score", fontsize=14)
    plt.plot(nor_result_only_sc.value_counts().index.tolist(), nor_result_only_sc.value_counts().values.tolist(), c='b', alpha=0.5, marker='o', ls='', label='正常序列')
    plt.plot(exp_result_only_sc.value_counts().index.tolist(), exp_result_only_sc.value_counts().values.tolist(), c='r', alpha=0.5, marker='v', ls='', label='异常序列')
    plt.xlabel("未见系统调用异常分数", fontproperties=font_set)
    plt.ylabel("周期样本数量", fontproperties=font_set)
    plt.legend(prop=font_set)
    create_tiff('UnSeenSc_OnlySc.tiff')
    print('analysis_unseen_sc() Success')

def analysis_pid_tid(dataloader, scenario_name):
    data_type_list = [
        {'type': TRAINING, 'data': dataloader.training_data()},
        {'type': VALIDATION, 'data': dataloader.validation_data()},
        {'type': TEST, 'data': dataloader.test_data()}
    ]

    # plt.figure(figsize=(18, 15))
    plt.title("PID TID Scatter")
    x_labels = ['PID Switch Freq', 'PID Unique', 'TID Switch Freq', 'TID Unique']

    for data_type in data_type_list:
        for recording in tqdm(data_type['data'],
                              f"P/Tid Load DataSet Array {data_type['type']}".rjust(27)):

            exploit_start_time = 0
            recording_type = dataloader._metadata_list[data_type['type']][recording.name]['recording_type']
            if recording_type == RecordingType.NORMAL_AND_ATTACK or recording_type == RecordingType.ATTACK:
                exploit_start_time = recording.metadata()["time"]["exploit"][0]["absolute"] * (10 ** 9)
                if exploit_start_time == 0:
                    print('!!!!!!!!!!!!! exploit_start_time! why it is 0')

            for (data_array, df) in recording.df_and_np():
                if data_array is None:
                    continue

                for data in data_array:
                    index = data[0]
                    y = data[41:45] / len(df[index])
                    if exploit_start_time == 0 or exploit_start_time > df[index].iloc[-1]['time']:
                        # plt.scatter(x_labels, y, c='b', alpha=0.5, s=6)
                        plt.plot(x_labels, y, c='b', alpha=0.5, marker='o', ls='', label='正常序列')
                    else:
                        plt.plot(x_labels, y, c='r', alpha=0.5, marker='v', ls='', label='异常序列')
                        # plt.scatter(x_labels, y, c='r', alpha=0.5, s=10)
    plt.xlabel("类型", fontproperties=font_set)
    plt.ylabel("频率", fontproperties=font_set)
    plt.legend(prop=font_set)
    create_tiff('PID_TID_Num_Freq.tiff')
    # plt.show()
    print('analysis_pid_tid() Success')

def analysis_time_delta(dataloader, scenario_name):
    # np.save(os.path.join(data_path, str(df.iloc[-1]['time']) + '.npy'), np.array(data))
    # train too much, so we only analysis validation and test
    data_type_list = [
        # {'type': TRAINING, 'data': dataloader.training_data()},
        {'type': VALIDATION, 'data': dataloader.validation_data()},
        {'type': TEST, 'data': dataloader.test_data()}
    ]

    normal_max_scNum = 0
    exploit_max_scNum = 0
    # plt.figure(figsize=(18, 15))
    plt.title("TimeDelta Scatter")
    x_labels = [i for i in range(0, 39)]

    for data_type in data_type_list:
        for recording in tqdm(data_type['data'],
                              f"Load DataSet Array {data_type['type']}".rjust(27)):

            exploit_start_time = 0
            recording_type = dataloader._metadata_list[data_type['type']][recording.name]['recording_type']
            if recording_type == RecordingType.NORMAL_AND_ATTACK or recording_type == RecordingType.ATTACK:
                exploit_start_time = recording.metadata()["time"]["exploit"][0]["absolute"] * (10 ** 9)
                if exploit_start_time == 0:
                    print('!!!!!!!!!!!!! exploit_start_time! why it is 0')

            for (data_array, df) in recording.df_and_np():
                if data_array is None:
                    continue

                for data in data_array:
                    index = data[0]
                    y = data[2:41]
                    number_line = len(df[index])
                    if exploit_start_time == 0 or exploit_start_time > df[index].iloc[-1]['time']:
                        # plt.scatter(x_labels, y, c='b', alpha=0.5, s=6)
                        plt.plot(x_labels, y, c='b', alpha=0.5, marker='o', ls='', label='正常序列')
                        normal_max_scNum = normal_max_scNum if normal_max_scNum > number_line else number_line
                    else:
                        # plt.scatter(x_labels, y, c='r', alpha=0.5, s=10)
                        plt.plot(x_labels, y, c='r', alpha=0.5, marker='v', ls='', label='异常序列')
                        exploit_max_scNum = exploit_max_scNum if exploit_max_scNum > number_line else number_line

    plt.xlabel("时间间隔", fontproperties=font_set)
    plt.ylabel("时间间隔数量", fontproperties=font_set)
    plt.legend(prop=font_set)
    create_tiff(os.path.join(DATAOUT_DIR, scenario_name, 'TimeDeltaScatter.tiff'))
    # plt.show()
    print('normal_max_scNum %d exploit_max_scNum %d ' % (normal_max_scNum, exploit_max_scNum))
    print('analysis_time_delta() Success')


def create_background_pic():
    year = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    loss = [21, 20, 25, 23, 30, 25, 32, 39, 66, 104, 105]

    plt.bar(year, loss)
    plt.xlabel("时间", fontproperties=font_set)
    plt.ylabel("网络攻击事件数量", fontproperties=font_set)
    plt.legend(prop=font_set)
    create_tiff(os.path.join(DATAOUT_DIR, 'background_pic.tiff'))

if __name__ == '__main__':

    LID_DS_VERSION_NUMBER = 1

    LID_DS_VERSION = [
            "LID-DS-2019",
            "LID-DS-2021"
            ]
    # scenarios ordered by training data size asc
    SCENARIOS = [
        "Juice-Shop",
        "CVE-2020-13942",
        "CWE-89-SQL-injection",
        "CVE-2012-2122",
        "CVE-2018-3760",
        "CVE-2020-23839",
        "CVE-2020-9484",
        "EPS_CWE-434",
        "CVE-2014-0160",
        "CVE-2017-7529",
        "Bruteforce_CWE-307",
        "CVE-2019-5418",
        "PHP_CWE-434",
    ]
    SCENARIO_RANGE = SCENARIOS[0:13]
    create_background_pic()
    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(work_dir,
                                     scenario_name)
        dataloader = dataloader_factory(scenario_path, direction=Direction.CLOSE)
        # 统计文件个数
        # get_file_num(dataloader, scenario_name)
        # analysis_unseen_sc(dataloader, scenario_name)
        # analysis_unseen_args(dataloader, scenario_name)
        # analysis_syscall_size(dataloader, scenario_name)
        analysis_ret_max(dataloader, scenario_name)
        # analysis_sc_max_freq(dataloader, scenario_name)
        # analysis_time_delta(dataloader, scenario_name)
        # analysis_pid_tid(dataloader, scenario_name)

        print('Success')