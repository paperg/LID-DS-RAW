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
# from dataloader.dateset_create_gp import index_timedelta_start, index_timdelta_end, index_ptid_start, index_ptid_end, index_uai, index_usi
from dataloader.syscall import Syscall

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

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
def plot_accuracies(accuracy_list, folder):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
    plt.savefig(f'plots/{folder}/training-graph.pdf')
    plt.clf()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranAD(nn.Module):
    def __init__(self, feats, lr=0.001, dropout=0.2, hid_times=4, nhead = 1, use_ae2=True):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self._use_ae2 = use_ae2
        self._hid_times = hid_times
        self.lr = lr

        self.n_feats = feats
        self._dropout = dropout

        # self.pos_encoder = PositionalEncoding(feats, 0.1, 1024)
        # encoder_layers1 = TransformerEncoderLayer(d_model=feats * 2, nhead=9, dim_feedforward=feats * 2, dropout=0.1)
        # self.transformer_encoder1 = TransformerEncoder(encoder_layers1, 1)

        encoder_layers2 = TransformerEncoderLayer(d_model=feats * 2, nhead=nhead, dim_feedforward=feats * 2, dropout=0.1)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, 1)
        # decoder_layers1 = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        # self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        # decoder_layers2 = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        # self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        # self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

        self.encoder1 = torch.nn.Sequential(

            torch.nn.Linear(feats * 2, feats * hid_times * 2),
            torch.nn.Dropout(p=self._dropout),
            torch.nn.SELU(),
            # torch.nn.Sigmoid(),
            # torch.nn.Tanh(),

            torch.nn.Linear(feats * hid_times * 2, feats * hid_times * 4),
            torch.nn.Dropout(p=self._dropout),
            # torch.nn.Tanh(),
            torch.nn.SELU(),
            # torch.nn.Sigmoid(),

            torch.nn.Linear(feats * hid_times * 4, feats * hid_times * 8),
            torch.nn.Dropout(p=self._dropout),
            torch.nn.SELU(),
            # torch.nn.Tanh(),
            # torch.nn.Sigmoid(),
        )

        self.decoder1 = torch.nn.Sequential(
            torch.nn.Linear(feats * hid_times * 8, feats * hid_times * 4),
            torch.nn.Dropout(p=self._dropout),
            torch.nn.SELU(),
            # torch.nn.Sigmoid(),

            torch.nn.Linear(feats * hid_times * 4, feats * hid_times),
            torch.nn.Dropout(p=self._dropout),
            torch.nn.SELU(),
            # torch.nn.Tanh(),
            # torch.nn.SELU(),
            # torch.nn.Sigmoid(),

            torch.nn.Linear(feats * hid_times, feats),
            torch.nn.Dropout(p=self._dropout),
            # torch.nn.Tanh(),
            torch.nn.SELU(),
            # torch.nn.Sigmoid(),
        )

        self.encoder2 = torch.nn.Sequential(
            torch.nn.Linear(feats * 2, feats * hid_times),
            torch.nn.Dropout(p=self._dropout),
            # torch.nn.SELU(),
            torch.nn.Tanh(),
            # torch.nn.Sigmoid(),

            torch.nn.Linear(feats * hid_times, feats * 2 * hid_times),
            torch.nn.Dropout(p=self._dropout),
            torch.nn.SELU(),
            # torch.nn.Tanh(),
            # torch.nn.Sigmoid(),

            # torch.nn.Linear(feats * 2 * hid_times * 2, feats * 2 * hid_times * 4),
            # torch.nn.Dropout(p=self._dropout),
            # torch.nn.SELU(),
        )

        # Building an decoder
        self.decoder2 = torch.nn.Sequential(

            torch.nn.Linear(feats * hid_times * 8, feats * hid_times * 4),
            torch.nn.Dropout(p=self._dropout),
            torch.nn.SELU(),
            # torch.nn.Tanh(),
            # torch.nn.Sigmoid(),

            torch.nn.Linear(feats * hid_times * 4, feats * hid_times),
            torch.nn.Dropout(p=self._dropout),
            torch.nn.SELU(),
            # torch.nn.Tanh(),
            # torch.nn.Sigmoid(),

            torch.nn.Linear(feats * hid_times, feats),
            torch.nn.Dropout(p=self._dropout),
            torch.nn.SELU()
            # torch.nn.Tanh(),
            # torch.nn.Sigmoid(),
        )

        self.norm = LayerNorm(feats * 2)

        for m in self.encoder1:
            if isinstance(m, nn.Linear):
                fan_in = m.in_features
                nn.init.normal_(m.weight, 0, math.sqrt(1. / fan_in))

        for m in self.decoder1:
            if isinstance(m, nn.Linear):
                fan_in = m.in_features
                nn.init.normal_(m.weight, 0, math.sqrt(1. / fan_in))

        for m in self.encoder2:
            if isinstance(m, nn.Linear):
                fan_in = m.in_features
                nn.init.normal_(m.weight, 0, math.sqrt(1. / fan_in))

        for m in self.decoder2:
            if isinstance(m, nn.Linear):
                fan_in = m.in_features
                nn.init.normal_(m.weight, 0, math.sqrt(1. / fan_in))
    def max_norm(self, max_val=2, eps=1e-8):
        for name, param in self.named_parameters():
            if 'bias' not in name:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, 0, max_val)
                param = param * (desired / (eps + norm))

    def forward(self, src):
        # src: (S, N, E)(S, N, E).
        # 其中S是源序列长度，T是目标序列长度，N是批处理大小，E是特征编号

        # encoder1 = self.encoder1(src)
        # reconstruct1 = self.decoder1(encoder1)
        # rec_loss1 = (reconstruct1 - src) ** 2
        # if self._use_ae2:
        #     ed2_inputs = torch.cat([src, rec_loss1], axis=2)
        #     encoder2 = self.transformer_encoder2(ed2_inputs)
        #     reconstruct2 = self.decoder2(encoder2)
        # else:
        #     reconstruct2 = None
        # self.max_norm()


        # ed2_inputs = torch.cat([src, last_loss], axis=2)
        # encoder1 = self.transformer_encoder1(ed2_inputs)

        # 先使用正常输入与重构误差 全 0 进行重构，得到重构误差 rec_loss
        rec_loss = torch.zeros_like(src)
        modle_input = torch.cat((src, rec_loss), dim=2)
        # modle_input = self.norm(modle_input)
        ae_input = self.transformer_encoder2(modle_input)
        encoder1 = self.encoder1(ae_input)
        reconstruct1 = self.decoder1(encoder1)
        rec_loss1 = (reconstruct1 - src) ** 2

        # 重构误差 rec_loss 不为零了 ，再传入模型，得到的误差可能会放大差异
        modle_input = torch.cat((src, rec_loss1), dim=2)
        ae_input = self.transformer_encoder2(modle_input)
        encoder2 = self.encoder1(ae_input)
        reconstruct2 = self.decoder2(encoder2)

        self.max_norm()

        return reconstruct1, reconstruct2, rec_loss1

class Transformer_ad(BuildingBlock):
    def __init__(self,
                 input_dim: int,
                 epochs=30,
                 dropout=0.1,
                 num_head=4,
                 hidden_layers=4,
                 batch_size=64,
                 use_dict:dict={},
                 model_path='Models/GPMODEL',
                 scenario_path='L:\\hids\\dataSet\\GP_DATA_DIR'):
        """
        Args:
            input_dim:          input dimension
            epochs:             set training epochs of LSTM
            hidden_layers:      amount of LSTM-layers
            hidden_dim:         dimension of LSTM-layer
            batch_size:         set maximum batch_size
            model_path:         path to save trained Net to
            force_train:        force training of Net
        """
        super().__init__()

        self._dependency_list = []
        # hyper parameter
        self._dropout = dropout
        self._input_dim = input_dim
        self._batch_size = batch_size
        self._epochs = epochs
        self._hidden_layers = hidden_layers
        self._num_head = num_head

        self._use_timedelta = use_dict['use_timedelta']
        self._use_ptidfreq = use_dict['use_ptidfreq']
        self._use_usa = use_dict['use_usa']
        self._use_usc = use_dict['use_usc']
        self._use_freq = use_dict['use_freq']
        self._use_sc_max_params = use_dict['use_sc_max_params']
        self._use_ret = use_dict['use_ret']
        self._use_ae2 = use_dict['mode_use_ae2']
        self.zscore_nor = StandardScaler()
        model_dir = os.path.split(model_path)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self._model_path = model_path
        print(f'model path {model_path}')
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
        self._transformer = None
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
        self.Net = TranAD(self._input_dim, lr=0.001, dropout=self._dropout, nhead = self._num_head, use_ae2 = self._use_ae2)
        self.Net.to(self._device)
        # self._last_loss = torch.zeros(1, 16, self._input_dim).to(self._device)
        # self._cal_last_loss = torch.zeros(1, 1, self._input_dim).to(self._device)
        self._loss = torch.nn.MSELoss()
        self._calloss = torch.nn.MSELoss(reduction='none')
        # ptid + usa + usc + freq + sc_max
        self.pid_tid_max = [0] * 100
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
                # self.pid_tid_max[0:normal_max_ptid_end] = np.fmax(self.pid_tid_max[0:normal_max_ptid_end],
                #                                                 np.max(recoding_data[0][:, index_ptid_start:index_ptid_end], axis=0))

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
            self._test_data['x'] = self.data_Normalization(self._test_data['x']).astype(np.float32)
            inputs = Variable(torch.Tensor(self._test_data['x'])).to(self._device)
            local_bs = inputs.shape[0]
            inputs = inputs.view(local_bs, -1, self._input_dim)
            window = inputs.permute(1, 0, 2)
            # if self._use_ae2:
            #     reconstruct, rec2_input, reconstruct_resc = self.Net(window)
            #     val_loss = self._calloss(rec2_input, reconstruct_resc)
            # else:
            #     # No AE 2
            #     reconstruct, rec2_input, _ =  self.Net(window)
            #     val_loss = self._calloss(reconstruct, window)

            reconstruct1, reconstruct2, rec_loss1 = self.Net(window)
            loss = self._calloss(window,reconstruct2)

            rec_loss1 = rec_loss1.view(-1, self._input_dim)
            test_loss = torch.mean(loss, dim=0)

            return test_loss.cpu().detach().numpy(), rec_loss1.cpu().detach().numpy(), self._test_data['x'], self._test_data['y']
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

        # df[:39] = df[:39] * 0.2
        # df[39:] = df[39:] * 0.8
        return data

    def _create_train_data(self, val: bool):
        print(f'pid_tid_max {self.pid_tid_max}')
        if not val:
            self._training_data['x'] = self.data_Normalization(self._training_data['x'], True).astype(np.float32)
            x_tensors = Variable(torch.Tensor(self._training_data['x'])).to(self._device)
            # y_tensors = Variable(torch.Tensor(self._training_data['y'])).to(self._device)
            # y_tensors = y_tensors.long()
            # x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            # print(f"Training Shape x: {x_tensors.shape} y: {y_tensors.shape}")
            print(f"Training Shape x: {x_tensors.shape}")
            return SyscallFeatureDataSetSigle(x_tensors)
        else:
            self._validation_data['x'] = self.data_Normalization(self._validation_data['x']).astype(np.float32)
            x_tensors = Variable(torch.Tensor(self._validation_data['x'])).to(self._device)
            # y_tensors = Variable(torch.Tensor(self._validation_data['y'])).to(self._device)
            # y_tensors = y_tensors.long()
            # x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Validation Shape x: {x_tensors.shape}")
            return SyscallFeatureDataSetSigle(x_tensors)

    def fit(self):
        """
        fit model only if it could not be loaded
        set model state.
        convert training data to tensors and feed into custom dataset
        call torch dataloader with prepared batch_indices
            needed because end of recording cuts a batch
        create actual net for fitting
        define hyperparameters, iterate through DataSet and train Net
        keep hidden and cell state over batches, only reset with new recording
        """
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
            optimizer = torch.optim.AdamW(self.Net.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

            for epoch in tqdm(range(self._epochs),
                              'training network:'.rjust(25),
                              unit=" epochs"):
                n = epoch + 1
                self.Net.train()
                for i, inputs in enumerate(train_dataloader, 0):
                    for j in range(0, 1):
                        # tmp_inputs = inputs + (time_mat_tentor[self.gp_pos] * j)
                        tmp_inputs = inputs
                        local_bs = tmp_inputs.shape[0]
                        tmp_inputs = tmp_inputs.view(local_bs, -1, self._input_dim)
                        window = tmp_inputs.permute(1, 0, 2)
                        reconstruct1, reconstruct2, rec_loss1= self.Net(window)
                        train_loss = (1 / n) * self._loss(window, reconstruct1) + (1 - 1 / n) * self._loss(window,  reconstruct2)
                        optimizer.zero_grad()
                        # calculates the loss of the loss function
                        train_loss.backward()
                        # improve from loss, i.e backpro, val_loss: %1.5fp
                        optimizer.step()
                scheduler.step()

                # reset hidden state
                self.Net.eval()
                val_loss = 0.0

                for inputs in val_dataloader:
                    # inputs, labels = data
                    local_bs = inputs.shape[0]
                    inputs = inputs.view(local_bs, -1, self._input_dim)
                    window = inputs.permute(1, 0, 2)
                    # if self._use_ae2:
                    #     reconstruct, rec2_input, reconstruct_resc = self.Net(window)
                    # else:
                    #     reconstruct, rec2_input, _= self.Net(window)

                    reconstruct1, reconstruct2, rec_loss1= self.Net(window)
                    train_loss = (1 / n) * self._loss(window, reconstruct1) + (1 - 1 / n) * self._loss(window, reconstruct2)
                    val_loss = self._loss(window, reconstruct2)

                    # val_loss = self._loss(window, reconstruct)

                print(f"Epoch: {epoch}, L1 {train_loss} val_loss: {val_loss}")

        else:
            print(f"Net already trained. Using model {self._model_path}")
            pass

    def _calculate(self, syscall: Syscall):
        """

        remove label from feature_list
        feed feature_list and hidden state into model.
        model returns probabilities of every syscall seen in training
        + index 0 for unknown syscall
        index of actual syscall gives predicted_prob
        1 - predicted_prob is anomaly score

        Returns:
            float: anomaly score

        """
        # feature_list = self.
        # .get_result(syscall)
        if syscall[0] is not None:
            data = self.get_recording_need_data(syscall, True)
            x_array = self.data_Normalization(data).astype(np.float32)

            x_tensor = torch.Tensor(x_array).to(self._device)
            inputs = x_tensor.view(1, -1, self._input_dim)
            window = inputs.permute(1, 0, 2)
            # if self._use_ae2:
            #     reconstruct, rec2_input, reconstruct_resc = self.Net(window)
            #     loss = self._calloss(rec2_input, reconstruct_resc).reshape(-1, self._input_dim)
            # else:
            #     reconstruct, rec2_input, _ = self.Net(window)
            #     loss = self._calloss(window, reconstruct).reshape(-1, self._input_dim)

            reconstruct1, reconstruct2, rec_loss1= self.Net(window)
            loss = self._calloss(window,reconstruct2).reshape(-1, self._input_dim)

            return loss.cpu().detach().numpy()
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