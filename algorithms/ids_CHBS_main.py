import sys
sys.path.append('.')
from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
import pandas as pd
import networkx as nx
from collections import Counter
from urllib.parse import urlparse
from string import digits
import os
import itertools
import time
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle
from rich.console import Console
from rich import print
from rich.table import Table
from dataloader.direction import Direction
from dataloader.dataloader_factory import dataloader_factory
from algorithms.persistance import save_to_mongo
from algorithms.features.impl.sequence_period import Sequence_period
from dataloader.data_loader_2021 import RecordingType
from algorithms.performance_measurement import Performance
from pprint import pprint
from algorithms.decision_engines.tranAD import Mine_AD
pd.options.mode.chained_assignment = None

OUTPUT_FILES_NAMES = ['seen_syscalls', 'seen_args', 'max_freq', 'thresh_list']
USN = 'USN'
UAN = 'UAN'
WEIGHT = 'weight'
MODE_SO = 'STAT/OPEN'
MODE_EX = 'EXECVE'
MODE_CL = 'CLONE'
EMPTY = ''

#syscalls
OPEN = 'open'
STAT = 'stat'
EXECVE = 'execve'
CLONE = 'clone'

ARGS_COL = 'Params'
SYSCALL_COL = 'syscall'
LOSS = 'loss'

ANOMALY = 'anomaly'
NORMAL =  'normal'

SYSCALLS_ARGS = ['open', 'stat', 'clone', 'execve']

PATH_LENGTH = 3
BETA_FOLD_INCREASE = 3
CHUNK_SIZE=20
PROCESS_NUM=8

EPOCH = 120
OPTIMIZER = 'Adam'
LOSS_FUNC = 'mse'
BATCH_SIZE = 50
VALIDATION_SPLIT = 0.2
ACTIVATION = 'sigmoid'
REG_RATE = 0.001
ENCODING_DIM = DECODING_DIM = 2
BOTTLENECK = 1
VERBOSE = 1
USE_CHBS=True
def _format_input(inputs, TRAINING_MODE=True):

    # inputs = list(itertools.chain.from_iterable(inputs))

    inputs = [x for x in inputs if x]
    ready_anomaly_vectors = np.array(inputs)
    return ready_anomaly_vectors[:,:3], ready_anomaly_vectors[:,-1]

class TrainModel(nn.Module):
    def __init__(self, input_dim):
        super(TrainModel, self).__init__()
        # L1
        self.L1 = nn.Linear(input_dim, ENCODING_DIM)
        self.L1_A = nn.Sigmoid()
        # L2
        self.L2 = nn.Linear(ENCODING_DIM, BOTTLENECK)
        self.L2_A = nn.Sigmoid()
        # L3
        self.L3 = nn.Linear(BOTTLENECK, BOTTLENECK)
        self.L3_A = nn.Sigmoid()
        # L4
        self.L4 = nn.Linear(BOTTLENECK, DECODING_DIM)
        self.L4_A = nn.Sigmoid()
        # out
        self.out = nn.Linear(DECODING_DIM, input_dim)
        self.out_a = nn.Sigmoid()

    def forward(self, input):
        # encoder
        L1 = self.L1_A(self.L1(input))
        L2 = self.L2_A(self.L2(L1))

        # decoder
        L3 = self.L3_A(self.L3(L2))
        L4 = self.L4_A(self.L4(L3))

        output = self.out_a(self.out(L4))

        return output

class Training:
    def __init__(self):
        # get model
        # CHIDS
        if USE_CHBS:
            self.model = TrainModel(3)
        else:
            self.max_array = [0] * 3
            self._calloss = torch.nn.MSELoss(reduction='none')
            # Mine Module
            # Mine_AD(self._input_dim, lr=0.001, dropout=self._dropout, nhead = self._num_head, use_ae2 = self._use_ae2)
            self.model = Mine_AD(3, lr=0.001, dropout=0.4, nhead = 3)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

    def _get_thresholds_list(self, model, inp):
        if USE_CHBS:
            thresh_list = []
            prediction = model(inp)
            prediction_l = prediction.reshape(prediction.shape[0], prediction.shape[2]).cpu().detach().numpy()
            inp_l = inp.reshape(inp.shape[0], inp.shape[2]).cpu().detach().numpy()
            training_reconstruction_errors = np.mean(np.square(prediction_l - inp_l), axis=1)

            for _theta in np.arange(0.2, 2.2, 0.2):
                thresh_list.append(max(training_reconstruction_errors) * _theta)
        else:
            window = inp.permute(1, 0, 2)
            reconstruct1, reconstruct2, rec_loss1 = model(window)
            loss = self._calloss(window,reconstruct2).reshape(-1, 3)
            thresh_list = loss.cpu().detach().numpy()
            thresh_list = thresh_list.max(axis=0)
        return thresh_list

    def val_model(self, val_anomaly_vectors):
        formatted_anomaly_vectors_val, _= _format_input(val_anomaly_vectors)
        val_anomaly_vectors = formatted_anomaly_vectors_val.reshape(formatted_anomaly_vectors_val.shape[0], 1,
                                                                     formatted_anomaly_vectors_val.shape[1])

        val_dataset = torch.Tensor(val_anomaly_vectors).to(self._device)
        self.model.eval()
        thresh_list = self._get_thresholds_list(self.model, val_dataset)
        if not USE_CHBS:
            if isinstance(thresh_list, np.ndarray):
                self.max_array = np.fmax(self.max_array, thresh_list)
            else:
                if thresh_list is not None:
                    self.max_array = max(thresh_list, self.max_array)
            thresh_list = self.max_array
        return thresh_list

    def train_model(self, anomaly_vectors):
        formatted_anomaly_vectors,_= _format_input(anomaly_vectors)
        training_anomaly_vectors = formatted_anomaly_vectors.reshape(formatted_anomaly_vectors.shape[0], 1, formatted_anomaly_vectors.shape[1])
        self.model.train()

        # train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_dataset = torch.Tensor(training_anomaly_vectors).to(self._device)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=BATCH_SIZE)

        for epoch in tqdm(range(EPOCH),
                          'training network:'.rjust(25),
                          unit=" epochs"):
            for i, data in enumerate(train_dataloader, 0):
                if USE_CHBS:
                    outputs = self.model(data)
                    optimizer.zero_grad()
                    classify_loss = criterion(outputs, data)

                    regularization_loss = 0
                    for param in self.model.parameters():
                        regularization_loss += torch.sum(abs(param))
                    loss = classify_loss + REG_RATE * regularization_loss
                else:
                    window = data.permute(1, 0, 2)
                    reconstruct1, reconstruct2, rec_loss1 = self.model(window)
                    loss = (1 / (epoch + 1)) * criterion(window, reconstruct1) + (1 - 1 / (epoch + 1)) * criterion(window, reconstruct2)
                    optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch: %d, loss: %1.5f\n" % (epoch, loss.item()))

        # model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNC)
        # model.fit(training_anomaly_vectors, training_anomaly_vectors, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)

    def _get_reconstruction_loss(self, scap_anomaly_vectors):
        formatted_anomaly_vectors, vector_time= _format_input(scap_anomaly_vectors)
        trace_anomaly_vectors = formatted_anomaly_vectors.reshape(formatted_anomaly_vectors.shape[0], 1, formatted_anomaly_vectors.shape[1])
        self.model.eval()
        input_tensor = torch.Tensor(trace_anomaly_vectors).to(self._device)
        if USE_CHBS:
            trace_prediction = self.model(input_tensor)
            trace_prediction_l = trace_prediction.reshape(trace_prediction.shape[0], trace_prediction.shape[2]).cpu().detach().numpy()
            trace_anomaly_vectors = trace_anomaly_vectors.reshape(trace_anomaly_vectors.shape[0], trace_anomaly_vectors.shape[2])
            errors = np.mean(np.square(trace_prediction_l - trace_anomaly_vectors), axis=1)
        else:
            window = input_tensor.permute(1, 0, 2)
            reconstruct1, reconstruct2, rec_loss1 = self.model(window)
            loss = self._calloss(window, reconstruct2)
            test_loss = torch.mean(loss, dim=0)
            errors = test_loss.cpu().detach().numpy(),

        return errors, vector_time
def get_filepath(raw_arg, category):

    if category == MODE_SO:
        file_path = raw_arg[5:]

    if category == MODE_EX:
        file_path = raw_arg[9:]

    if category == MODE_CL:
        file_path = raw_arg[4:]

    return process_path(file_path)
def process_path(file_path_i):

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
class SSGraph:
    def __init__(self, sequence_data, seen_syscalls, seen_args):
        self.sequence_data = sequence_data
        self.seen_syscalls = seen_syscalls
        self.seen_args = seen_args
        self.ssg_edges = []

    def _get_graph(self, g_edges):
        graph = nx.DiGraph((x, y, {WEIGHT: v}) for (x, y), v in Counter(g_edges).items())
        return graph

    def _get_graph_size(self, graph):
        return graph.size(weight=WEIGHT)

    def _get_usi(self, distinct_unseen_syscalls, sequence_graph):

        if distinct_unseen_syscalls:
            usn_in_centrality = nx.in_degree_centrality(sequence_graph).get(USN)
            usn_out_centrality = nx.out_degree_centrality(sequence_graph).get(USN)

            if isinstance(usn_in_centrality, type(None)):
                usn_in_centrality = 0

            if isinstance(usn_out_centrality, type(None)):
                usn_out_centrality = 0

            return len(distinct_unseen_syscalls) * (usn_in_centrality + usn_out_centrality)

        else:
            return 0

    def _get_uai(self, distinct_unseen_args, distinct_syscalls_with_args, sequence_graph):

        if distinct_unseen_args:
            uan_in_centrality = nx.in_degree_centrality(sequence_graph).get(UAN)
            uan_out_centrality = nx.out_degree_centrality(sequence_graph).get(UAN)

            if isinstance(uan_in_centrality, type(None)):
                uan_in_centrality = 0

            if isinstance(uan_out_centrality, type(None)):
                uan_out_centrality = 0

            return len(distinct_syscalls_with_args) * len(distinct_unseen_args) * (uan_in_centrality + uan_out_centrality)

        else:
            return 0

    def get_ssg_features(self):

        # get distinct unseen syscalls
        distinct_unseen_syscalls = set(self.sequence_data.loc[~self.sequence_data.syscall.isin(self.seen_syscalls)][SYSCALL_COL].tolist())

        # aggregate all unseen syscalls under the USN node
        self.sequence_data.loc[~self.sequence_data.syscall.isin(self.seen_syscalls), SYSCALL_COL] = USN

        # process the arguments of open, stat, execve, and clone syscalls
        self.sequence_data.loc[self.sequence_data.syscall.isin([USN]), ARGS_COL] = EMPTY
        # self.sequence_data.loc[self.sequence_data.syscall.isin([STAT, OPEN]), ARGS_COL] = \
        #     self.sequence_data[ARGS_COL].str[1].dropna().apply(get_filepath, args=(MODE_SO,))
        # self.sequence_data.loc[self.sequence_data.syscall.isin([EXECVE]), ARGS_COL] = \
        #     self.sequence_data[ARGS_COL].str[0].dropna().apply(get_filepath, args=(MODE_EX,))
        # self.sequence_data.loc[self.sequence_data.syscall.isin([CLONE]), ARGS_COL] = \
        #     self.sequence_data[ARGS_COL].str[1].dropna().apply(get_filepath, args=(MODE_CL,))

        # get distinct syscalls (open, stat, clone, execve) that involve unseen args
        distinct_syscalls_with_unseen_args = set(
            self.sequence_data.loc[(self.sequence_data.syscall.isin(SYSCALLS_ARGS)) & self.sequence_data.Params.notnull()
                    & (~self.sequence_data.Params.isin(self.seen_args))][SYSCALL_COL].tolist())

        # aggregate all unseen arguments under the UAN node
        self.sequence_data.loc[(self.sequence_data.syscall.isin(SYSCALLS_ARGS)) & self.sequence_data.Params.notnull() &
            (~self.sequence_data.Params.isin(self.seen_args)), SYSCALL_COL] = UAN

        # get distinct unseen arguments
        distinct_unseen_args = set(self.sequence_data.loc[(self.sequence_data.syscall.isin([UAN]))][ARGS_COL].tolist())

        # prepare the SSG edges
        syscalls_list = self.sequence_data[SYSCALL_COL].tolist()
        for i in range(len(syscalls_list) - 1):
            self.ssg_edges.append((syscalls_list[i], syscalls_list[i + 1]))


        ssg = self._get_graph(self.ssg_edges)

        # get the SSG features
        # usi : unseen syscalls importance
        # uai : unseen arguments importance
        usi = self._get_usi(distinct_unseen_syscalls, ssg)
        uai = self._get_uai(distinct_unseen_args, distinct_syscalls_with_unseen_args, ssg)
        ssg_size = self._get_graph_size(ssg)

        return usi, uai, ssg_size

class AnomalyVector:

    def __init__(self, traces, seen_syscalls, seen_args, max_freq_syscalls):
        self.traces = traces
        self.seen_syscalls = seen_syscalls
        self.seen_args = seen_args
        self.max_s = max_freq_syscalls


    def construct_anomaly_vector(self, trace):
        anomaly_vectors = []
        for sequence in trace:
            if sequence.empty:
                anomaly_vectors.append([])
            else:
                graph_ob = SSGraph(sequence, self.seen_syscalls, self.seen_args)
                usi, uai, ssg_size = graph_ob.get_ssg_features()
                fi = ssg_size / (self.max_s * BETA_FOLD_INCREASE)
                anomaly_vector = [fi, usi, uai, sequence['time'].max()]
                anomaly_vectors.append(anomaly_vector)

        return anomaly_vectors


    def get_anomaly_vectors(self):
        # pool = Pool(processes=PROCESS_NUM)
        # all_anomaly_vectors = pool.map(self.construct_anomaly_vector, self.traces, chunksize=CHUNK_SIZE)
        all_anomaly_vectors = []
        for one in self.traces:
            all_anomaly_vectors.extend(self.construct_anomaly_vector(one))
        return all_anomaly_vectors

class SeenSyscalls:

    def __init__(self, scaps_dfs):
        self.scaps_dfs = scaps_dfs

    def _syscalls_per_scap(self, df_all):
        result = set()
        # for key, df in df_all_dict.items():
        for df in df_all:
            result = result | set(list(df[SYSCALL_COL]))

        return result

    def seen_syscalls(self):
        # pool = Pool(processes=PROCESS_NUM)
        # all_syscalls = pool.map(self._syscalls_per_scap, self.scaps_dfs, chunksize=CHUNK_SIZE)
        all_syscalls = set()
        for one in self.scaps_dfs:
            all_syscalls = all_syscalls | self._syscalls_per_scap(one)
        # seen_syscalls = set(list(itertools.chain.from_iterable(all_syscalls)))
        seen_syscalls = set(all_syscalls)
        print(f'Seen syscall num {len(seen_syscalls)}')
        del all_syscalls
        return list(seen_syscalls)
class SeenArgs:

    def __init__(self, scaps_dfs):
        self.scaps_dfs = scaps_dfs

    def _get_args(self, df_all):
        result = set()
        for scap_df in df_all:
            syscalls_with_args = scap_df[[SYSCALL_COL, ARGS_COL]]
            # syscalls_with_args.loc[~syscalls_with_args.syscall.isin(SYSCALLS_ARGS), ARGS_COL] = EMPTY
            #
            # syscalls_with_args.loc[syscalls_with_args.syscall.isin([STAT, OPEN]), ARGS_COL] = \
            #     syscalls_with_args[ARGS_COL].dropna().apply(get_filepath, args=(MODE_SO,))
            #
            # syscalls_with_args.loc[syscalls_with_args.syscall.isin([CLONE]), ARGS_COL] = \
            #     syscalls_with_args[ARGS_COL].dropna().apply(get_filepath, args=(MODE_CL,))
            #
            # syscalls_with_args.loc[syscalls_with_args.syscall.isin([EXECVE]), ARGS_COL] = \
            #     syscalls_with_args[ARGS_COL].dropna().apply(get_filepath, args=(MODE_EX,))
            #
            # list_of_args = syscalls_with_args.loc[syscalls_with_args.syscall.isin(SYSCALLS_ARGS)
            #                 & syscalls_with_args.Params.notnull()][ARGS_COL].tolist()

            list_of_args = syscalls_with_args.loc[syscalls_with_args.syscall.isin(SYSCALLS_ARGS)
                            & syscalls_with_args.Params.notnull()][ARGS_COL].tolist()

            result = result | set(list_of_args)
        return result

    def seen_args(self):
        # self._get_args(self.scaps_dfs)
        all_args = set()
        # pool = Pool(processes=PROCESS_NUM)
        # all_args = pool.map(self._get_args, self.scaps_dfs, chunksize=CHUNK_SIZE)
        for one in self.scaps_dfs:
            all_args = all_args | self._get_args(one)
        # seen_args = set(list(itertools.chain.from_iterable(all_args)))
        seen_args = set(all_args)
        del all_args
        return list(seen_args)

def load_pickled_file(file):
    with open(file, 'rb') as f:
        output = pickle.load(f)
    return output

def save_file(training_elements, model, output_folder):
    console = Console()

    try:
        path = output_folder
        if not Path(path).is_dir():
            print('Create dir %s' %path)
            os.mkdir(path)
        if USE_CHBS:
            torch.save(model.state_dict(), path+'/'+'model.h5')
            console.print('model' + ' -------->  saved successfully in {}'.format(output_folder), style='green bold')

        for _, ele in enumerate(training_elements):
            with open(output_folder + "/" + OUTPUT_FILES_NAMES[_] + ".pkl", 'wb') as f:
                pickle.dump(ele, f)

            console.print(OUTPUT_FILES_NAMES[_] + ' -------->  saved successfully in {}'.format(output_folder), style='green bold')

    except OSError:
        print("Creation of the directory %s failed" % os.getcwd())

class CHBS_Model(BuildingBlock):
    def __init__(self, model_path):
        super().__init__()
        self.anomaly_vectors = None
        self.val_anomaly_vectors = None
        self.seen_syscalls = None
        self.seen_args = None
        self.max_freq = 0
        self.len_scaps = 0
        self.train_pro = Training()
        self.model_save_path = model_path
        self.need_train = True

        # torch.save(model.state_dict(), path + '/' + 'model.h5')
        model_save_path = model_path + '/' + 'model.h5'

        if os.path.exists(model_save_path):
            try:
                if USE_CHBS:
                    self.need_train = False
                    self.train_pro.model.load_state_dict(torch.load(model_save_path))
                    print(f'Load Model in {model_save_path}')
                #  ['seen_syscalls', 'seen_args', 'max_freq', 'thresh_list']
                self.seen_syscalls = load_pickled_file(model_path + '/seen_syscalls.pkl')
                self.seen_args = load_pickled_file(model_path + '/seen_args.pkl')
                self.max_freq = load_pickled_file(model_path + '/max_freq.pkl')
                self.thresh_list = load_pickled_file(model_path + '/thresh_list.pkl')
            except:
                print('Load Model and Data Failed')
                return


    def depends_on(self):
        return []

    def need_train(self):
        return self.need_train

    def train_on(self, scaps_dfs):
        if not self.need_train:
            return
        if scaps_dfs is not None:
            self.len_scaps = len(scaps_dfs)
            print("prepare SCAP Files ready")
            self.seen_syscalls = self._seen_syscalls(scaps_dfs)
            print("prepare seen_syscalls ready")
            self.seen_args = self._seen_args(scaps_dfs)
            print("prepare seen_args ready")
            self.max_freq = self._get_max_seq_freq(scaps_dfs)
            print("prepare max_freq ready")

            anomaly_vectors= self._get_anomaly_vectors(scaps_dfs, self.seen_syscalls, self.seen_args, self.max_freq)
            self.train_pro.train_model(anomaly_vectors)

    def val_on(self, scaps_dfs):
        if scaps_dfs is not None:
            print("prepare Val anomaly vector ready")
            val_anomaly_vectors = self._get_anomaly_vectors(scaps_dfs, self.seen_syscalls, self.seen_args, self.max_freq)
            self.thresh_list = self.train_pro.val_model(val_anomaly_vectors)
            return self.thresh_list
    def fit(self):
        output_table = Table(title='------------ Training Summary ----------')
        output_table.add_column("Number of training scaps", style="magenta")
        output_table.add_column("previously seen syscalls", style="magenta")
        output_table.add_column("previously seen arguments", style="magenta")
        output_table.add_column("Thresholds", style="magenta")
        output_table.add_row(str(self.len_scaps), str(self.seen_syscalls)[1:-1], str(self.seen_args)[1:-1], str(self.thresh_list)[1:-1])

        print(output_table)
        if self.need_train and USE_CHBS:
            save_file([self.seen_syscalls, self.seen_args, self.max_freq, self.thresh_list], self.train_pro.model, self.model_save_path)

    def _calculate(self, scaps_dfs):
        return scaps_dfs

    def test_on(self, scaps_dfs):
        if scaps_dfs is not None:
            # print("prepare Test anomaly vector ready")
            test_anomaly_vectors = self._get_anomaly_vectors(scaps_dfs, self.seen_syscalls, self.seen_args, self.max_freq)
            result, vector_time= self._classify(test_anomaly_vectors)
            del test_anomaly_vectors
            return result, vector_time


    def _classify(self, scap_anomaly_vectors):
        result_of_thetas = []
        if USE_CHBS:
            result = pd.DataFrame()
            result[LOSS], vector_time= self.train_pro._get_reconstruction_loss(scap_anomaly_vectors)
            for threshold in self.thresh_list:
                result[ANOMALY] = result[LOSS] > threshold
                result[NORMAL] = result[LOSS] <= threshold
                # is_anomalous = True if len(result[result[ANOMALY] == True]) > 0 else False

                result_of_thetas.append(result)
        else:
            result, vector_time = self.train_pro._get_reconstruction_loss(scap_anomaly_vectors)
            arr = np.greater(self.thresh_list * 2, result[0])
            for a in arr:
                if np.all(a):
                    result_of_thetas.append(False)
                else:
                    result_of_thetas.append(True)

        return result_of_thetas, vector_time
    def _seen_syscalls(self, scaps_dfs):
        ss_obj = SeenSyscalls(scaps_dfs)
        seen_syscalls = ss_obj.seen_syscalls()
        return seen_syscalls

    def _seen_args(self, scaps_dfs):
        sa_obj = SeenArgs(scaps_dfs)
        seen_args = sa_obj.seen_args()

        return seen_args

    def _get_max_seq_freq(self, traces):
        merged_list_sequences = list(itertools.chain.from_iterable(traces))
        max_seq_freq =  max([len(sequence) for sequence in merged_list_sequences])
        # for trace in traces:
        #     for key, scap_df in trace.items():
        #         max_seq_freq = max(max_seq_freq, len(scap_df))
        return max_seq_freq

    def _get_anomaly_vectors(self, traces, seen_syscalls, seen_args, max_seq_freq):
        av_obj = AnomalyVector(traces, seen_syscalls, seen_args, max_seq_freq)
        all_anomaly_vectors = av_obj.get_anomaly_vectors()
        return all_anomaly_vectors


if __name__ == '__main__':

    LID_DS_VERSION_NUMBER = 1
    LID_DS_VERSION = [
        "LID-DS-2019",
        "LID-DS-2021"
    ]

    # scenarios ordered by training data size asc
    SCENARIOS = [
        "ZipSlip",
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

        "CVE-2017-12635_6"
    ]

    SCENARIO_RANGE = SCENARIOS[0:1]

    # getting the LID-DS base path from argument or environment variable
    LID_DS_BASE_PATH = 'L:/hids'

    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(LID_DS_BASE_PATH,
                                     "dataSet",
                                     scenario_name)

        dataloader = dataloader_factory(scenario_path, direction=Direction.CLOSE)
        seq_dfs = Sequence_period()
        model = CHBS_Model(os.path.join('K:/hids/Models', scenario_name))
        # performance = Performance(False)
        perf_list = []
        for i in range(10):
            perf_list.append(Performance(False))

        # Training
        if model.need_train:
            for recording in tqdm(dataloader.training_data(),
                                  f"{scenario_name} train data ".rjust(27),
                                  unit=" recording"):
                for df_one_file in recording.period_df():
                    if df_one_file is not None:
                        seq_dfs.train_on(list(df_one_file.values()))

            df_all_list = seq_dfs._calculate(None)
            model.train_on(df_all_list)

            # Validation
            seq_dfs.new_recording()

            for recording in tqdm(dataloader.validation_data(),
                                  f"{scenario_name} val data ".rjust(27),
                                  unit=" recording"):
                for df_one_file in recording.period_df():
                    if df_one_file is not None:
                        seq_dfs.train_on(list(df_one_file.values()))

            df_all_list = seq_dfs._calculate(None)
            model.val_on(df_all_list)

        model.fit()

        # Test
        seq_dfs.new_recording()
        exploit_time_list = []
        start = time.time()
        for recording in tqdm(dataloader.test_data(),
                              f"{scenario_name} test data ".rjust(27),
                              unit=" recording"):

            exploit_start_time = 0
            recording_type = dataloader._metadata_list['test'][recording.name]['recording_type']
            if recording_type == RecordingType.NORMAL_AND_ATTACK or recording_type == RecordingType.ATTACK:
                exploit_start_time = recording.metadata()["time"]["exploit"][0]["absolute"] * (10 ** 9)
                if exploit_start_time == 0:
                    print('!!!!!!!!!!!!! exploit_start_time! why it is 0')
            seq_dfs.new_recording()
            for df_one_file in recording.period_df():
                if df_one_file is not None:
                    exploit_time_list.append(exploit_start_time)
                    seq_dfs.train_on(list(df_one_file.values()))
            df_all_list = seq_dfs._calculate(None)
            if len(df_all_list) > 0:
                is_anomaly, vector_time = model.test_on(df_all_list)
                if USE_CHBS:
                    for i, performance in enumerate(perf_list):
                        performance.new_recording(recording)

                    for i, performance in enumerate(perf_list):
                        for index, cur_time in enumerate(vector_time):
                            need_hanle, current_exploit_time = performance.analyze_batchs(cur_time * (10 ** (-9)), is_anomaly[i]['anomaly'][index])

                    save_result = None
                    for i, performance in enumerate(perf_list):
                        results = performance.get_results()
                        results['scenario'] = scenario_name
                        results['Model'] = 'CHBS_Model'
                        # pprint(results)
                        if save_result is None:
                            save_result = results
                        if results['recall'] > save_result['recall']:
                            save_result = results
                else:
                    performance = perf_list[0]
                    performance.new_recording(recording)
                    for index, cur_time in enumerate(vector_time):
                        need_hanle, current_exploit_time = performance.analyze_batchs(cur_time * (10 ** (-9)),
                                                                                      is_anomaly[index])


        # results = performance.get_results()
        # results['scenario'] = scenario_name
        # results['Model'] = 'CHBS_Model'

        # save_result = results
        end = time.time()
        detection_time = (end - start) / 60  # in min
        save_result['date'] = str(datetime.now().date())
        save_result['detection_time'] = detection_time

        save_to_mongo(save_result)