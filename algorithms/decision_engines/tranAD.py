import math
import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.autograd import Variable

from tqdm import tqdm

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall

import matplotlib.pyplot as plt


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
    def __init__(self, feats, lr):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class Transformer_ad(BuildingBlock):
    def __init__(self,
                 input_vector: BuildingBlock,
                 distinct_syscalls: int,
                 input_dim: int,
                 epochs=30,
                 dropout=0.1,
                 num_head=4,
                 hidden_layers=4,
                 batch_size=64,
                 model_path='Models/Transformer',
                 force_train=False):
        """
        Args:
            distinct_syscalls:  amount of distinct syscalls in training data
            input_dim:          input dimension
            epochs:             set training epochs of LSTM
            hidden_layers:      amount of LSTM-layers
            hidden_dim:         dimension of LSTM-layer
            batch_size:         set maximum batch_size
            model_path:         path to save trained Net to
            force_train:        force training of Net
        """
        super().__init__()
        self._input_vector = input_vector
        self._dependency_list = [input_vector]
        self._distinct_syscalls = distinct_syscalls + 1

        # hyper parameter
        self._dropout = dropout
        self._input_dim = input_dim
        self._batch_size = batch_size
        self._epochs = epochs
        self._hidden_layers = hidden_layers
        self._num_head = num_head

        model_dir = os.path.split(model_path)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self._model_path = model_path
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

        self._train_loss = np.array([[0, 0, 0]])

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Net = TranAD(self._input_dim, lr=0.001)
        self.loss = torch.nn.MSELoss(reduction='none')
        self.Net.to(self._device)

    def forward(self, X):
        return self.Net(X)

    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall):
        """

        create training data and keep track of batch indices
        batch indices are later used for creation of batches

        Args:
            feature_list (int): list of prepared features for DE

        """
        feature_list = self._input_vector.get_result(syscall)
        if feature_list is not None:
            x = np.array(feature_list)
            y = feature_list[-self._input_dim:]
            self._training_data['x'].append(x)
            self._training_data['y'].append(y)
            self._current_batch.append(self._batch_counter)
            self._batch_counter += 1
            if len(self._current_batch) == self._batch_size:
                self._batch_indices.append(self._current_batch)
                self._current_batch = []
        else:
            pass

    def val_on(self, syscall: Syscall):
        """

        create validation data and keep track of batch indices
        batch indices are later used for creation of batches

        Args:
            feature_list (int): list of prepared features for DE

        """
        feature_list = self._input_vector.get_result(syscall)
        if feature_list is not None:
            x = np.array(feature_list)
            y = feature_list[-self._input_dim:]
            self._validation_data['x'].append(x)
            self._validation_data['y'].append(y)
            self._current_batch_val.append(self._batch_counter_val)
            self._batch_counter_val += 1
            if len(self._current_batch_val) == self._batch_size:
                self._batch_indices_val.append(self._current_batch_val)
                self._current_batch_val = []
        else:
            pass

    def test_on(self, syscall: Syscall):
        feature_list = self._input_vector.get_result(syscall)
        if feature_list is not None:
            x = np.array(feature_list)
            y = feature_list[-self._input_dim:]
            self._test_data['x'].append(x)
            self._test_data['y'].append(y)
            self._current_batch_test.append(self._batch_counter_test)
            self._batch_counter_test += 1
            if len(self._current_batch_test) == self._batch_size:
                self._batch_indices_test.append(self._current_batch_test)
                self._current_batch_test = []

            return True
        else:
            return None
    def cal_test_loss(self):
        result_loss = np.array([[0,0,0]])
        x_tensors = Variable(torch.Tensor(self._test_data['x'])).to(self._device)
        y_tensors = Variable(torch.Tensor(self._test_data['y'])).to(self._device)
        print(f"Test Shape x: {x_tensors.shape} y: {y_tensors.shape}")
        test_dataset = SyscallFeatureDataSet(x_tensors, y_tensors)
        test_dataloader = DataLoader(test_dataset, batch_sampler=self._batch_indices_test)
        for inputs,labels in tqdm(test_dataloader):
            local_bs = inputs.shape[0]
            inputs = inputs.view(local_bs, -1, self._input_dim)
            window = inputs.permute(1, 0, 2)

            elem = labels.view(1, local_bs, self._input_dim)
            z = self.Net(window, elem)
            if isinstance(z, tuple): z = z[1]
            loss = self.loss(z, elem)
            vl = loss.cpu().detach().numpy().reshape(-1, self._input_dim)
            result_loss = np.concatenate((result_loss, vl), axis=0)
            # loss_mean = torch.mean(loss)
            # print('loss_mean %f' % loss_mean)
        return result_loss
    def _create_train_data(self, val: bool):
        if not val:
            x_tensors = Variable(torch.Tensor(self._training_data['x'])).to(self._device)
            y_tensors = Variable(torch.Tensor(self._training_data['y'])).to(self._device)
            # y_tensors = y_tensors.long()
            # x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Training Shape x: {x_tensors.shape} y: {y_tensors.shape}")
            return SyscallFeatureDataSet(x_tensors, y_tensors), y_tensors
        else:
            x_tensors = Variable(torch.Tensor(self._validation_data['x'])).to(self._device)
            y_tensors = Variable(torch.Tensor(self._validation_data['y'])).to(self._device)
            # y_tensors = y_tensors.long()
            # x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Validation Shape x: {x_tensors.shape} y: {y_tensors.shape}")
            return SyscallFeatureDataSet(x_tensors, y_tensors), y_tensors

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

        if self._state == 'build_training_data':
            self._state = 'fitting'
        if self.Net is not None:
            train_dataset, y_tensors = self._create_train_data(val=False)
            val_dataset, y_tensors_val = self._create_train_data(val=True)
            # for custom batches
            train_dataloader = DataLoader(train_dataset, batch_sampler=self._batch_indices)
            val_dataloader = DataLoader(val_dataset, batch_sampler=self._batch_indices_val)

            accuracy_list = []
            # Net hyperparameters
            optimizer = torch.optim.AdamW(self.Net.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

            for epoch in tqdm(range(self._epochs),
                              'training network:'.rjust(25),
                              unit=" epochs"):
                n = epoch + 1
                l1s = []
                self.Net.train()
                for i, data in enumerate(train_dataloader, 0):
                    inputs, labels = data
                    local_bs = inputs.shape[0]
                    inputs = inputs.view(local_bs, -1, self._input_dim)
                    window = inputs.permute(1, 0, 2)
                    # elem = window[-1, :, :].view(1, local_bs, self._input_dim)
                    # def forward(self, X, Image_X)
                    elem = labels.view(1, local_bs, self._input_dim)
                    z = self.Net(window, elem)
                    l1 = self.loss(z, elem) if not isinstance(z, tuple) else (1 / n) * self.loss(z[0], elem) + (
                                1 - 1 / n) * self.loss(z[1],
                                                       elem)
                    if isinstance(z, tuple): z = z[1]
                    l1s.append(torch.mean(l1).item())
                    loss = torch.mean(l1)

                    optimizer.zero_grad()
                    # calculates the loss of the loss function
                    loss.backward(retain_graph=True)
                    # improve from loss, i.e backpro, val_loss: %1.5fp
                    optimizer.step()
                scheduler.step()
                acc = np.mean(l1s)
                accuracy_list.append((acc, optimizer.param_groups[0]['lr']))

                # reset hidden state
                self.new_recording()
                self.Net.eval()
                val_loss = 0.0
                for data in val_dataloader:
                    inputs, labels = data
                    local_bs = inputs.shape[0]
                    inputs = inputs.view(local_bs, -1, self._input_dim)
                    window = inputs.permute(1, 0, 2)

                    elem = labels.view(1, local_bs, self._input_dim)
                    z = self.Net(window, elem)
                    if isinstance(z, tuple): z = z[1]
                    val_loss = self.loss(z, elem)
                    vl = val_loss.cpu().detach().numpy().reshape(-1, self._input_dim)
                    self._train_loss = np.concatenate((self._train_loss, vl), axis=0)
                    loss = torch.mean(val_loss)
                print(f"Epoch: {epoch}, L1 {acc} val_loss: {loss}")

            plot_accuracies(accuracy_list, 'TranAD')
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
        feature_list = self._input_vector.get_result(syscall)
        if feature_list:
            x_tensor = Variable(torch.Tensor(np.array([feature_list]))).to(self._device)
            inputs = x_tensor.view(1, -1, self._input_dim)
            elem = x_tensor[-self._input_dim:].view(1, -1, self._input_dim)
            z = self.Net(inputs, elem)
            if isinstance(z, tuple): z = z[1]
            loss = self.loss(z, elem)[0]
            return loss.cpu().detach().numpy()
        else:
            return None

    def _accuracy(self, outputs, labels):
        """

        calculate accuracy of last epoch

        """
        hit = 0
        miss = 0
        for i in range(len(outputs) - 1):
            if outputs[i] == labels[i]:
                hit += 1
            else:
                miss += 1
        return hit / (hit + miss)

    def get_results(self, results):
        '''
        self._dropout = dropout
        self._input_dim = input_dim
        self._batch_size = batch_size
        self._epochs = epochs
        self._hidden_layers = hidden_layers
        self._num_head = num_head
        '''
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