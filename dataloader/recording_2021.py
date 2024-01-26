import os
import csv
import json
import pcapkit
import zipfile
import pandas as pd
import numpy as np
import pickle
from dataloader.base_recording import BaseRecording

from dataloader.direction import Direction
from dataloader.syscall import Syscall
from dataloader.resource_statistic import ResourceStatistic
from dataloader.syscall_2021 import Syscall2021

DATAOUT_DIR='K:/hids/GP_DATA_DIR'

class Recording2021(BaseRecording):
    """

        Single recording captured in 4 ways
        class provides functions to handle every type of recording
            --> syscall text file
            --> pcap packets
            --> json describing recording
            --> statistics of resources

        Args:
        path (str): path of recording
        name (str): name of file without extension

    """

    def __init__(self, path: str, name: str, direction: Direction):
        """

            Save name and path of recording.

            Parameter:
            path (str): path of associated files
            name (str): name without path and extension

        """
        self.path = path
        self.name = name
        self._direction = direction
        self.check_recording()
        self.path_list = self.path.split('\\')

    def syscalls(self) -> str:
        """

            Prepare stream of syscalls,
            yield single lines

            Returns:
            str: syscall text line

        """
        try:
            with zipfile.ZipFile(self.path, 'r') as zipped:
                with zipped.open(self.name + '.sc') as unzipped:
                    for line_id, syscall in enumerate(unzipped, start=1):
                        syscall_object = Syscall2021(self.path, syscall.decode('utf-8').rstrip(), line_id=line_id)
                        if self._direction != Direction.BOTH:
                            if syscall_object.direction() == self._direction:
                                yield syscall_object
                        else:
                            yield syscall_object

        except Exception:
            raise Exception(f'Error while working with file: {self.name} at {self.path}')


    def syscalls_df(self) -> str:
        """

            Prepare stream of syscalls,
            yield single lines

            Returns:
            str: syscall text line

        """
        try:
            with zipfile.ZipFile(self.path, 'r') as zipped:
                # try: engine='python',
                with zipped.open(self.name + '.sc') as unzipped:
                    df = pd.read_csv(unzipped, delim_whitespace=True, index_col=False, error_bad_lines=False, names=['time','UserID', 'PID', 'ProcessName', 'TID', 'syscall', 'DIR', 'RET', 'ARGS1', 'ARGS2', 'ARGS3', 'ARGS4'])

                # except Exception as e:
                #     print('Frist Read Error')
                #     with zipped.open(self.name + '.sc') as unzipped:
                #         df = pd.read_csv(unzipped, delim_whitespace=True, index_col=False, skiprows=1,
                #                          error_bad_lines=False,
                #                          names=['time', 'UserID', 'PID', 'ProcessName', 'TID', 'syscall', 'DIR', 'RET',
                #                                 'ARGS'])
                    # df.time = pd.to_datetime(df['time'])
                    # for period_time, period_cont in df.resample('900L', on='time'):
                    yield df

        except Exception:
            raise Exception(f'Error while working with file: {self.name} at {self.path}')

    def df_and_np(self) -> tuple:
        try:
            if 'test' in self.path:
                file_path = os.path.join(DATAOUT_DIR, self.path_list [-4], self.path_list [-3], self.path_list [-2], self.name)
            else:
                file_path = os.path.join(DATAOUT_DIR, self.path_list [-3], self.path_list [-2], self.name)
            array_file = os.path.join(file_path, 'array.npy')
            df_file = os.path.join(file_path, 'df_all.pkl')
            arr_all = None
            df_all = None
            if os.path.exists(array_file):
                arr_all = np.load(array_file, allow_pickle=True)
            else:
                print(f'{array_file} is not exists')

            if os.path.exists(df_file):
                with open(df_file, 'rb') as f:
                    df_all = pickle.load(f)
            else:
                print(f'{df_file} is not exists')

            yield tuple([arr_all, df_all])
            # files = glob.glob(file_path + '/FinalData/*.npy')
            # if len(files) == 0:
            #     print(f'files is none, {file_path}')
            # for file in files:
            #     datas = np.load(file)
            #     df_file = file.replace('FinalData', 'DataFrame').replace('npy', 'pkl')
            #     df = pd.read_pickle(df_file)
            #     yield tuple([datas,df])
        except Exception as ex:
            raise Exception(f'Error while get data working with file: {self.name} at {self.path}, {ex}')

    def period_df(self):
        try:
            if 'test' in self.path:
                file_path = os.path.join(DATAOUT_DIR, self.path_list [-4], self.path_list [-3], self.path_list [-2], self.name)
            else:
                file_path = os.path.join(DATAOUT_DIR, self.path_list [-3], self.path_list [-2], self.name)

            df_file = os.path.join(file_path, 'df_all.pkl')
            df_all = None

            if os.path.exists(df_file):
                with open(df_file, 'rb') as f:
                    df_all = pickle.load(f)
            else:
                print(f'{df_file} is not exists')

            yield df_all

        except Exception as ex:
            raise Exception(f'Error while get data working with file: {self.name} at {self.path}, {ex}')

    def packets(self):
        """

            Unzip and extract pcap objects,

            Returns:
            pcap obj: return pypcap Extractor object
            src:
                https://pypcapkit.jarryshaw.me/en/latest/foundation/extraction.html#pcapkit.foundation.extraction.Extractor

        """
        try:
            with zipfile.ZipFile(self.path, 'r') as zipped:
                file_list = zipped.namelist()
                for file in file_list:
                    if file.endswith('.pcap'):
                        zipped.extract(file, 'tmp')
            obj = pcapkit.extract(fin=f'tmp/{self.name}.pcap',
                                  engine='pyshark',
                                  store=True,
                                  nofile=True)
        except Exception:
            print(f'Error extracting pcap file {self.name}')
            return None
        finally:
            os.remove(f'tmp/{self.name}.pcap')

        return obj

    def resource_stats(self) -> list:
        """

            Read .res file of recording.
            Includes usage of following resources for a point in time:
                timestamp,
                cpu_usage,
                memory_usage,
                network_received,
                network_send,
                storage_read,
                storage_written

            Returns:
            List of used resources

        """
        statistics = []
        with zipfile.ZipFile(self.path, 'r') as zipped:
            with zipped.open(self.name + '.res') as unzipped:
                string = unzipped.read().decode('utf-8')
                reader = csv.reader(string.split('\n'), delimiter=',')
                # remove header
                next(reader)
                for row in reader:
                    if len(row) > 0:
                        statistics.append(ResourceStatistic(row))
        return statistics

    def metadata(self) -> dict:
        """

            Read json file and extract metadata as dict
            with following format:
            {"container": [
                    "ip": str,
                    "name": str,
                    "role": str
             "exploit": bool,
             "exploit_name": str,
             "image": str,
             "recording_time": int,
             "time":{
                    "container_ready": {
                        "absolute": float,
                        "source": str
                    },
                    "exploit": [
                        {
                            "absolute": float,
                            "name": str,
                            "source": str
                        }
                    ]
                    "warmup_end": {
                        "absolute": float,
                        "source": str
                    }
                }
            }

            Returns:
            dict: metadata dictionary

        """
        with zipfile.ZipFile(self.path, 'r') as zipped:
            with zipped.open(self.name + '.json') as unzipped:
                unzipped_byte_json = unzipped.read()
                unzipped_json = json.loads(unzipped_byte_json.decode('utf-8').replace("'", '"'))
        return unzipped_json

    def check_recording(self) -> bool:
        """

            check if zip file exists and if all necessary files are included

            Returns:
            bool: if check was succesfull

        """
        try:
            if not os.path.isfile(self.path):
                raise Exception(f'Missing .zip file for recording: {self.path}')
            with zipfile.ZipFile(self.path, 'r') as zipped:
                file_list = zipped.namelist()
                err_str = 'Recording Error: '
                if len(file_list) != 4:
                    if self.name + '.res' not in file_list:
                        res_err = 'Missing .res file '
                        err_str += res_err
                    if self.name + '.sc' not in file_list:
                        sc_err = 'Missing .sc file '
                        err_str += sc_err
                    if self.name + '.pcap' not in file_list:
                        pcap_err = 'Missing .pcap file '
                        err_str += pcap_err
                    if self.name + '.json' not in file_list:
                        json_err = 'Missing .json file '
                        err_str += json_err
                    if not os.path.isfile('missing_files.txt'):
                        with open('missing_files.txt', 'w+') as file:
                            file.write(err_str + f'in recording: {self.path}. \n')
                    else:
                        with open('missing_files.txt', 'a') as file:
                            file.write(err_str + f'in recording: {self.path}. \n')
                    print(f'{err_str}')
                    print('Have a look in missing_files.txt file')
        except Exception:
            print(f'Error with file {self.name} at {self.path}')
            if not os.path.isfile('missing_files.txt'):
                with open('missing_files.txt', 'w+') as file:
                    file.write(err_str + f'in recording: {self.path}. \n')
            else:
                with open('missing_files.txt', 'a') as file:
                    file.write(err_str + f'in recording: {self.path}. \n')
