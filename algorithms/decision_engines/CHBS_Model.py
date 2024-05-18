import math
import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, LayerNorm
from torch.nn import TransformerDecoder
from torch.autograd import Variable
from tqdm import tqdm
from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from algorithms.ids_CHBS_main import TrainModel

DATA_USED_BY_MODEL_DIR = 'data_used_by_model'

index_timedelta_start = 2
index_timdelta_end = 41
index_ptid_start = 41
index_ptid_end = 45
index_uai = 45
index_usi = 46
index_ret_total = 47
index_sc_max_start = 48
index_sc_max_end = 62

normal_max_ptid_end = index_ptid_end - index_ptid_start
normal_max_uai_end  = normal_max_ptid_end + 1
normal_max_usi_end  = normal_max_uai_end + 1
normal_max_freq_end = normal_max_usi_end + 1
normal_sc_max_freq_end = normal_max_freq_end + 1

ENCODING_DIM = DECODING_DIM = 4
BOTTLENECK = 2
#
# class Trainmodel(nn.Module):
#     def __init__(self, input_dim):
#         super(Trainmodel, self).__init__()
#         self.input_dim = input_dim
#         self._dropout = 0.2
#         hid_times = 4
#         self.encoder1 = torch.nn.Sequential(
#
#             torch.nn.Linear(input_dim, input_dim * hid_times),
#             torch.nn.Dropout(p=self._dropout),
#             torch.nn.SELU(),
#             # torch.nn.Sigmoid(),
#             # torch.nn.Tanh(),
#
#             torch.nn.Linear(input_dim * hid_times, input_dim * hid_times * 4),
#             torch.nn.Dropout(p=self._dropout),
#             # torch.nn.Tanh(),
#             torch.nn.SELU(),
#             # torch.nn.Sigmoid(),
#
#             torch.nn.Linear(input_dim * hid_times * 4, input_dim * hid_times * 8),
#             torch.nn.Dropout(p=self._dropout),
#             torch.nn.SELU(),
#             # torch.nn.Tanh(),
#             # torch.nn.Sigmoid(),
#         )
#
#         self.decoder1 = torch.nn.Sequential(
#             torch.nn.Linear(input_dim * hid_times * 8, input_dim * hid_times * 4),
#             torch.nn.Dropout(p=self._dropout),
#             torch.nn.SELU(),
#             # torch.nn.Sigmoid(),
#
#             torch.nn.Linear(input_dim * hid_times * 4, input_dim * hid_times),
#             torch.nn.Dropout(p=self._dropout),
#             torch.nn.SELU(),
#             # torch.nn.Tanh(),
#             # torch.nn.SELU(),
#             # torch.nn.Sigmoid(),
#
#             torch.nn.Linear(input_dim * hid_times, input_dim),
#             torch.nn.Dropout(p=self._dropout),
#             # torch.nn.Tanh(),
#             torch.nn.SELU(),
#             # torch.nn.Sigmoid(),
#         )
#
#         for m in self.encoder1:
#             if isinstance(m, nn.Linear):
#                 fan_in = m.in_features
#                 nn.init.normal_(m.weight, 0, math.sqrt(1. / fan_in))
#
#         for m in self.decoder1:
#             if isinstance(m, nn.Linear):
#                 fan_in = m.in_features
#                 nn.init.normal_(m.weight, 0, math.sqrt(1. / fan_in))
#     def max_norm(self, max_val=2, eps=1e-8):
#         for name, param in self.named_parameters():
#             if 'bias' not in name:
#                 norm = param.norm(2, dim=0, keepdim=True)
#                 desired = torch.clamp(norm, 0, max_val)
#                 param = param * (desired / (eps + norm))
#     def forward(self, input):
#         # encoder
#         encoder_out = self.encoder1(input)
#         output = self.decoder1(encoder_out)
#         return output

class CHBS_ad(BuildingBlock):
    def __init__(self,
                 input_dim: int,
                 epochs=30,
                 batch_size=64,
                 use_dict: dict = {},
                 model_path='Models/GPMODEL',
                 scenario_path='L:\\hids\\dataSet\\GP_DATA_DIR'):

        super().__init__()

        self._dependency_list = []
        # hyper parameter
        self._input_dim = input_dim
        self._batch_size = batch_size
        self._epochs = epochs
        self._use_timedelta = use_dict['use_timedelta']
        self._use_ptidfreq = use_dict['use_ptidfreq']
        self._use_usa = use_dict['use_usa']
        self._use_usc = use_dict['use_usc']
        self._use_freq = use_dict['use_freq']
        self._use_sc_max_params = use_dict['use_sc_max_params']
        self._use_ret = use_dict['use_ret']
        self._use_ae2 = use_dict['mode_use_ae2']
        self._module_is_mine = False

        model_dir = os.path.split(model_path)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self._model_path = model_path
        print(f'CHBS model path {model_path}')
        self._training_data = {
            'x': [],
            'y': []
        }
        self._validation_data = {
            'x': [],
            'y': []
        }
        self._test_data = {
            'x': [],
            'y': []
        }

        self._state = 'build_training_data'
        self._batch_indices = []
        self._batch_indices_val = []
        self._current_batch = []
        self._current_batch_val = []
        self._batch_indices_test = []
        self._current_batch_test = []
        self._batch_counter = 0
        self._batch_counter_val = 0
        self._batch_counter_test = 0

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Net = TrainModel(self._input_dim)
        self.Net.to(self._device)

        self._loss = torch.nn.MSELoss()
        # ptid + usa + usc + freq + sc_max
        self.pid_tid_max = [0] * 64
        self._need_prepare_data = True
        self._dataset_dir = scenario_path
        if not os.path.exists(self._dataset_dir):
            os.mkdir(self._dataset_dir)
        else:
            file_path = os.path.join(self._dataset_dir, 'val.npy')
            if os.path.exists(file_path):
                print('Data Used By model exist')
                self._need_prepare_data = False

    def depends_on(self):
        return self._dependency_list

    def get_dataset_directly(self):
        return self._need_prepare_data == False

    def get_recording_need_data(self, recoding_data, test: bool = False):
        # normal_max_ptid_end normal_max_uai_end normal_max_usi_end
        tmp_x = []
        if self._use_timedelta:
            tmp_x = recoding_data[0][:, index_timedelta_start:index_timdelta_end]
            # no Need normal

        if self._use_ptidfreq:
            colum = []
            for index, array_data in enumerate(recoding_data[0]):
                colum.append(array_data[index_ptid_start:index_ptid_end] / len(recoding_data[1][index]))

            if len(tmp_x) > 0:
                tmp_x = np.hstack((tmp_x, colum))
            else:
                tmp_x = colum
            if not test:
                self.pid_tid_max[0:normal_max_ptid_end] = [0 ,0 ,0 ,0]

        if self._use_usa:
            if len(tmp_x) > 0:
                tmp_x = np.hstack((tmp_x, recoding_data[0][:, index_uai:index_uai + 1]))
            else:
                tmp_x = recoding_data[0][:, index_uai:index_uai + 1]
            if not test:
                self.pid_tid_max[normal_max_ptid_end] = max(
                    self.pid_tid_max[normal_max_ptid_end], recoding_data[0][:, index_uai].max())

        if self._use_usc:
            if len(tmp_x) > 0:
                tmp_x = np.hstack((tmp_x, recoding_data[0][:, index_usi:index_usi + 1]))
            else:
                try:
                    tmp_x = recoding_data[0][:, index_usi:index_usi + 1]
                except:
                    print('Wrong Data')
                    return []
            if not test:
                self.pid_tid_max[normal_max_uai_end] = max(
                    self.pid_tid_max[normal_max_uai_end], recoding_data[0][:, index_usi].max())

        if self._use_ret:
            if len(tmp_x) > 0:
                tmp_x = np.hstack((tmp_x, recoding_data[0][:, index_ret_total:index_ret_total + 1]))
            else:
                try:
                    tmp_x = recoding_data[0][:, index_ret_total:index_ret_total + 1]
                except:
                    print('Wrong Data')
                    return []
            if not test:
                self.pid_tid_max[normal_max_freq_end] = max(
                    self.pid_tid_max[normal_max_freq_end], recoding_data[0][:, index_ret_total].max())

        if self._use_freq:
            freq_colum = np.array([0] * len(recoding_data[0]))
            for i in range(len(recoding_data[0])):
                freq_colum[i] = len(recoding_data[1][i])
            freq_colum = freq_colum.reshape(-1, 1)
            if len(tmp_x) > 0:
                tmp_x = np.hstack((tmp_x, freq_colum))
            else:
                tmp_x = freq_colum

            if not test:
                self.pid_tid_max[normal_max_usi_end] = max(
                    self.pid_tid_max[normal_max_usi_end], freq_colum.max())

        if self._use_sc_max_params:
            # sc_max_freq = np.array([0] * 8)
            if len(tmp_x) > 0:
                # 48 49 50 51 52 53 54 55 56
                # 1  2  3  4  5  6  7  8  9
                tmp_x = np.hstack((tmp_x, recoding_data[0][:, index_sc_max_start:53]))
                tmp_x = np.hstack((tmp_x, recoding_data[0][:, 54:56]))
                tmp_x = np.hstack((tmp_x, recoding_data[0][:, 57:index_sc_max_end]))
            else:
                # tmp_x = recoding_data[0][:, index_sc_max_start:index_sc_max_end]
                tmp_x = recoding_data[0][:, index_sc_max_start:53]
                tmp_x = recoding_data[0][:, 54:56]
                tmp_x = np.hstack((tmp_x, recoding_data[0][:, 57:index_sc_max_end]))
            if not test:
                self.pid_tid_max[normal_sc_max_freq_end:normal_sc_max_freq_end + 14] = np.maximum(
                    self.pid_tid_max[normal_sc_max_freq_end:normal_sc_max_freq_end + 14],
                    np.max(recoding_data[0][:, index_sc_max_start:index_sc_max_end], axis=0))

        return tmp_x

    def train_on(self, syscall):
        if self._need_prepare_data:
            if syscall[0] is not None:
                data = self.get_recording_need_data(syscall)
                for per_second_data in data:
                    self._training_data['x'].append(per_second_data)
                    self._current_batch.append(self._batch_counter)
                    self._batch_counter += 1
                    if len(self._current_batch) == self._batch_size:
                        self._batch_indices.append(self._current_batch)
                        self._current_batch = []
            else:
                pass
        else:
            file_path = os.path.join(self._dataset_dir, 'train.npy')
            batch_indeces_file_path = os.path.join(self._dataset_dir, 'train_batIndic.npy')
            self._training_data['x'] = np.load(file_path, allow_pickle=True).astype(np.float32)
            self._batch_indices = np.load(batch_indeces_file_path, allow_pickle=True)

            file_path = os.path.join(self._dataset_dir, 'data_normal_max.npy')
            self.pid_tid_max = np.load(file_path, allow_pickle=True)

    def val_on(self, syscall):
        if self._need_prepare_data:
            if syscall[0] is not None:
                data = self.get_recording_need_data(syscall)
                for per_second_data in data:
                    self._validation_data['x'].append(per_second_data)
                    # self._validation_data['y'].append(y)
                    self._current_batch_val.append(self._batch_counter_val)
                    self._batch_counter_val += 1
                    if len(self._current_batch_val) == self._batch_size:
                        self._batch_indices_val.append(self._current_batch_val)
                        self._current_batch_val = []
            else:
                pass
        else:
            file_path = os.path.join(self._dataset_dir, 'val.npy')
            batch_indeces_file_path = os.path.join(self._dataset_dir, 'val_batIndic.npy')
            self._validation_data['x'] = np.load(file_path, allow_pickle=True).astype(np.float32)
            self._batch_indices_val = np.load(batch_indeces_file_path, allow_pickle=True)

    def _save_dataset_to_file(self):
        if self._need_prepare_data:
            file_path = os.path.join(self._dataset_dir, 'data_normal_max.npy')
            np.save(file_path, self.pid_tid_max, allow_pickle=True)

            file_path = os.path.join(self._dataset_dir, 'train.npy')
            batch_indeces_file_path = os.path.join(self._dataset_dir, 'train_batIndic.npy')
            np.save(file_path, self._training_data['x'], allow_pickle=True)
            np.save(batch_indeces_file_path, self._batch_indices, allow_pickle=True)

            file_path = os.path.join(self._dataset_dir, 'val.npy')
            batch_indeces_file_path = os.path.join(self._dataset_dir, 'val_batIndic.npy')
            np.save(file_path, self._validation_data['x'], allow_pickle=True)
            np.save(batch_indeces_file_path, self._batch_indices_val, allow_pickle=True)

    def get_input_result(self, syscall):
        if syscall[0] is not None:
            data = self.get_recording_need_data(syscall, True)
            for index, per_second_data in enumerate(data):
                self._test_data['x'].append(per_second_data)
                self._test_data['y'].append(int(syscall[0][index][1]) * (10 ** (-9)))

                self._batch_counter_test += 1
                if self._batch_counter_test == self._batch_size:
                    self._batch_counter_test = 0

            return True
        else:
            return False

    def test_batch_finish(self):
        self._test_data['x'] = []
        self._test_data['y'] = []

    def cal_test_result(self):
        if len(self._test_data['x']) > 0:
            self.Net.eval()
            self._test_data['x'] = self.data_Normalization(self._test_data['x']).astype(np.float32)
            inputs = Variable(torch.Tensor(self._test_data['x'])).to(self._device)

            trace_prediction = self.Net(inputs)
            trace_prediction_l = trace_prediction.cpu().detach().numpy()
            inputs = inputs.cpu().detach().numpy()
            errors = np.mean(np.square(trace_prediction_l - inputs), axis=1)

            return errors, errors, self._test_data['x'], self._test_data['y']
        else:
            return None, None

    def data_Normalization(self, data, is_train=False):
        # if is_train:
        #     res = self.zscore_nor.fit_transform(data)
        # else:
        #     res = self.zscore_nor.transform(data)
        data = np.array(data)
        index = 0
        if self._use_timedelta:
            index += 39

        if self._use_ptidfreq:
            for i in range(index, index + 4):
                data[:, i] = data[:, i] / (self.pid_tid_max[i - index] + 1)
            index += 4

        if self._use_usa:
            data[:, index] = data[:, index] / (self.pid_tid_max[4] + 1)
            index += 1

        if self._use_usc:
            data[:, index] = data[:, index] / (self.pid_tid_max[5] + 1)
            index += 1

        if self._use_ret:
            data[:, index] = data[:, index] / (self.pid_tid_max[6] + 1)
            index += 1

        if self._use_freq:
            data[:, index] = data[:, index] / (self.pid_tid_max[7] + 1)
            index += 1

        if self._use_sc_max_params:
            for i in range(14):
                if i == 8 or i == 5:
                    continue
                if i > 8:
                    j = i - 2
                elif i > 5:
                    j = i - 1
                else:
                    j = i

                data[:, index + j] = data[:, index + j] / (self.pid_tid_max[8 + i] + 1)

        return data

    def _create_train_data(self, val: bool):
        print(f'pid_tid_max {self.pid_tid_max}')
        if not val:
            self._training_data['x'] = self.data_Normalization(self._training_data['x'], True).astype(np.float32)
            x_tensors = Variable(torch.Tensor(self._training_data['x'])).to(self._device)
            print(f"Training Shape x: {x_tensors.shape}")
            return SyscallFeatureDataSetSigle(x_tensors)
        else:
            self._validation_data['x'] = self.data_Normalization(self._validation_data['x']).astype(np.float32)
            x_tensors = Variable(torch.Tensor(self._validation_data['x'])).to(self._device)
            print(f"Validation Shape x: {x_tensors.shape}")
            return SyscallFeatureDataSetSigle(x_tensors)

    def fit(self):
        self._save_dataset_to_file()
        if self._state == 'build_training_data':
            self._state = 'fitting'
        if self.Net is not None:
            train_dataset = self._create_train_data(val=False)
            val_dataset = self._create_train_data(val=True)
            # for custom batches
            train_dataloader = DataLoader(train_dataset, batch_sampler=self._batch_indices)
            val_dataloader = DataLoader(val_dataset, batch_sampler=self._batch_indices_val)

            # Net hyperparameters
            optimizer = torch.optim.Adam(self.Net.parameters(), lr=0.001)
            for epoch in tqdm(range(self._epochs),
                              'training network:'.rjust(25),
                              unit=" epochs"):
                n = epoch + 1
                self.Net.train()
                for i, inputs in enumerate(train_dataloader, 0):
                    outputs = self.Net(inputs)
                    optimizer.zero_grad()
                    classify_loss = self._loss(outputs, inputs)
                    regularization_loss = 0
                    for param in self.Net.parameters():
                        regularization_loss += torch.sum(abs(param))

                    loss = classify_loss + 0.001 * regularization_loss
                    loss.backward()
                    optimizer.step()

                print(f"Epoch: {epoch}, Train Loss: {loss}")

        else:
            print(f"Net already trained. Using model {self._model_path}")
            pass

    def _calculate(self, syscall: Syscall):
        self.Net.eval()
        if syscall[0] is not None:
            data = self.get_recording_need_data(syscall, True)
            x_array = self.data_Normalization(data).astype(np.float32)

            inputs = torch.Tensor(x_array).to(self._device)
            prediction = self.Net(inputs)
            prediction_l = prediction.cpu().detach().numpy()
            inp_l = inputs.cpu().detach().numpy()
            training_reconstruction_errors = np.mean(np.square(prediction_l - inp_l), axis=1)

            return max(training_reconstruction_errors)
        else:
            return None


    def get_results(self, results):
        results['dropout'] = self._dropout
        results['epochs'] = self._epochs
        results['batch_size'] = self._batch_size
        results['input_dim'] = self._input_dim
        results['hidden_layers'] = self._hidden_layers
        results['num_head'] = self._num_head

        return results

class SyscallFeatureDataSet(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match length of Y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y


class SyscallFeatureDataSetSigle(Dataset):

    def __init__(self, X):
        self.X = X
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]