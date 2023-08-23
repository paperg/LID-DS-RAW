

import os
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from string import digits
import json

from dataloader.syscall import Syscall

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall

# open : open < fd=13(<f>/etc/apache2/.htpasswd) name=file_name
# stat  : stat < res=0 path=/var/www/private/upload.php
# clone : clone < res=466 exe=/usr/sbin/apache2
# execve:
def get_filepath(row):
    raw_arg = row['Params']
    if row['syscallName'] == 'open':
        file_path = raw_arg['name']
    elif row['syscallName'] == 'stat':
        file_path = raw_arg['path']
    elif row['syscallName'] == 'clone':
        file_path = raw_arg['exe']
    elif row['syscallName'] == 'execve':
        file_path = raw_arg['exe']

    return process_path(file_path)

def one_get_filepath(syscallName, raw_arg):

    if syscallName == 'open':
        file_path = raw_arg['name']
    elif syscallName == 'stat':
        file_path = raw_arg['path']
    elif syscallName == 'clone':
        file_path = raw_arg['exe']
    elif syscallName == 'execve':
        file_path = raw_arg['exe']

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


# open : open < fd=13(<f>/etc/apache2/.htpasswd) name=file_name
# stat  : stat < res=0 path=/var/www/private/upload.php
# clone : clone < res=466 exe=/usr/sbin/apache2
# execve:

# 以下都会调用 open， 可不需要
# write: write > fd=13(<f>/tmp/phpg5JQGg) size=4816
# read : read > fd=13(<f>/etc/apache2/.htpasswd) size=2096
# close : close > fd=13(<f>/etc/apache2/.htpasswd)
# fstat : fstat > fd=13(<f>/var/www/private/upload.php)
SYSCALLS_ARGS_FILE = ['open', 'stat', 'clone', 'execve']


class UnseenArgs():
    def __init__(self, is_training_data: bool = True,  result_save_path: str = './output'):
        super().__init__()
        # parameter
        self._is_training_data = is_training_data

        self._result_save_path = result_save_path
        if self._is_training_data:
            self._result_dict = {}
        else:
            self._seq_dict = {}
            self._seq_list = []
            self._seq_list_node = []
            self._seq_df = pd.DataFrame(columns=['ProcessName','syscallName', 'syscallInt', 'Ret', 'Params'])
            # load seen args into self._result_dict
            with open(os.path.join(self._result_save_path, 'All.json')) as f:
                self._result_dict = json.load(f)
            for key in self._result_dict:
                self._result_dict[key] = set(self._result_dict[key])
            print(f'Load result_dict from {self._result_save_path}/All.json')
    def record(self, syscall: Syscall, syscallARet):
        if syscall.name() in SYSCALLS_ARGS_FILE:
            args_list = one_get_filepath(syscall.name(), syscall.params())

            if self._is_training_data:
                if syscall.process_name() not in self._result_dict:
                    self._result_dict[syscall.process_name()] = set()

                self._result_dict[syscall.process_name()].add(args_list)
            else:

                self._seq_df.loc[len(self._seq_df.index)] = [syscall.process_name(), syscall.name(), syscallARet[0], syscallARet[1], args_list]
                # if syscall.process_name() not in self._seq_dict:
                #     self._seq_dict[syscall.process_name()] = []
                #
                # self._seq_dict[syscall.process_name()].append(args_list)
                # # 删除的时候，根据程序名，找到对应 list 删除头部即可
                # self._seq_list.append(syscall.process_name())
                # self._seq_list_node.append((syscallARet[0], syscallARet[1]))
    def End(self):
        if self._is_training_data:
            if len(self._result_dict) == 0:
                print('self._result_dict empty, return')
                return

            for key in self._result_dict:
                self._result_dict[key] = list(self._result_dict[key])
            with open(os.path.join(self._result_save_path, 'All.json'), 'w') as f:
                json.dump(self._result_dict, f, indent=4, ensure_ascii=True, sort_keys=False)
    def get_seen_sc_args(self, processName:str):
        if processName in self._result_dict:
            return self.self._result_dict[processName]

        return None

    def get_unseen_sc_args(self, seq_df):
        result = {}
        for pname in self._seq_df['ProcessName'].unique():
            arg_lists = set(self._seq_df[self._seq_df['ProcessName'] == pname]['Params'].tolist())
            result[pname] = arg_lists - self._result_dict[pname]

        return result

    def remove_by_index(self, seq_df, end):
        if self._is_training_data is not True:
            for row in seq_df[:end][['ProcessName', 'syscallName']].itertuples(index=True, name='Pandas'):
                scname = getattr(row, "syscallName")
                if scname in SYSCALLS_ARGS_FILE:
                    pname = getattr(row, "ProcessName")
                    if self._seq_df.iloc[0]['ProcessName'] != pname or self._seq_df.iloc[0]['syscallName'] != scname:
                        print(f"Some thing error, remove by index, {self._seq_df.iloc[0]['ProcessName']} != {pname}, {self._seq_df.iloc[0]['syscallName']} != {scname}")
                    self._seq_df.drop(0, inplace=True)
                    self._seq_df.reset_index(drop=True, inplace=True)


#
# class UnseenArgs():
#     def __init__(self, is_training_data: bool = True,  result_save_path: str = './output'):
#         super().__init__()
#         # parameter
#         self._is_training_data = is_training_data
#
#         self._result_save_path = result_save_path
#         if self._is_training_data:
#             self._result_dict = {}
#         else:
#             # load seen args into self._result_dict
#             self._result_dict = np.load(os.path.join(self._result_save_path, 'All.npy'), allow_pickle=True)
#             print(f'Load result_dict from {self._result_save_path}/All.npy')
#     def record(self, seq_df):
#         if seq_df is not None:
#             syscalls_with_args = seq_df[['ProcessName', 'syscallName', 'Params']]
#             # syscalls_with_args.loc[~syscalls_with_args.syscallName.isin(SYSCALLS_ARGS_FILE), 'Params'] = ''
#             # get syscalls_with_args ProcessName list
#             pnames = syscalls_with_args['ProcessName'].unique()
#             for name in pnames:
#                 df = syscalls_with_args[syscalls_with_args['ProcessName'] == name]
#                 list_of_args = df.loc[
#                     df.syscallName.isin(SYSCALLS_ARGS_FILE)].dropna().apply(
#                     get_filepath, axis=1)
#
#                 # if empty ,skip
#                 if list_of_args.empty:
#                     continue
#
#                 if self._is_training_data:
#                      if name not in self._result_dict:
#                         self._result_dict[name] = set(list_of_args)
#                      else:
#                          self._result_dict[name] = self._result_dict[name]  | (set(list_of_args))
#                 else:
#                     np.save(os.path.join(self._result_save_path, str(seq_df['time'].max()) , name), np.unique(list_of_args))
#
#
#     def End(self):
#         if self._is_training_data:
#             np.save(os.path.join(self._result_save_path, 'All.npy'), self._result_dict)
#     def get_seen_sc_args(self, processName:str):
#         if self._is_training_data:
#             return self.self._result_dict[processName]
#
#         return None
#
#     def get_unseen_sc_args(self, seq_df):
#         result = {}
#         if seq_df is not None:
#             syscalls_with_args = seq_df[['ProcessName', 'syscallName', 'Params']]
#             syscalls_with_args.loc[~syscalls_with_args.syscallName.isin(SYSCALLS_ARGS_FILE), 'Params'] = ''
#             # get syscalls_with_args ProcessName list
#             pnames = syscalls_with_args['ProcessName'].unique()
#             for name in pnames:
#                 df = syscalls_with_args[syscalls_with_args['ProcessName'] == name]
#                 list_of_args = df.loc[
#                     df.syscallName.isin(SYSCALLS_ARGS_FILE)].dropna().apply(
#                     get_filepath, axis=1)
#
#                 if list_of_args.empty:
#                     continue
#
#                 if name not in self._result_dict:
#                     print(f'{name} is not in _result_dict')
#                 else:
#                     result[name] = set(list_of_args) - self._result_dict[name]
#
#         return result

