import math
import os
import torch

import numpy as np
from torch import nn
from tqdm import tqdm
from dataloader.syscall import Syscall
from algorithms.building_block import BuildingBlock
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class AttentionDecoder(Decoder):
    """The base attention-based decoder interface.

    Defined in :numref:`sec_seq2seq_attention`"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionalEncoding(nn.Module):
    """Positional encoding.

    Defined in :numref:`sec_self-attention-and-positional-encoding`"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


#@save
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    """Transformer编码器"""
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        # self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        # X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        X = self.pos_encoding(X * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class ImageNet(nn.Module):

    def __init__(self, num_hiddens=64,
                 num_layers=2,
                 dropout=0.1,
                 num_heads=8,
                 input_dim=64,
                 ):
        super().__init__()

        ffn_num_input, ffn_num_hiddens = input_dim, input_dim * 2
        key_size, query_size, value_size = input_dim, input_dim, input_dim
        norm_shape = [input_dim]

        self.transformer_encode = TransformerEncoder(key_size, query_size, value_size, num_hiddens, norm_shape,
                                                     ffn_num_input,
                                                     ffn_num_hiddens, num_heads, num_layers, dropout)

        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, 1, dropout)

        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

        self.dense = nn.Linear(num_hiddens * 2, 61)

    def forward(self, X , Image_X):
        # 1. normal for Image_X
        K = Image_X / 0xff

        # 2. encoder for X using trnasfromer encode mode
        Q = self.transformer_encode(X, None)

        Q = Q.permute(1, 0, 2)
        # 3. calculate the attention for Q and K
        V = self.attention(Q, K, K, None)
        Y = self.addnorm1(Q, V)
        V = self.addnorm2(Y, self.ffn(Y))

        input= torch.cat((Q, V), dim=2)
        # 4. using V as input for liner layer to output label
        out = nn.ReLU(inplace=True)(self.dense(input))

        return out.view(out.shape[1], -1)


class ImageIDS(BuildingBlock):
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

        self._training_name = {}
        self._validation_name = {}

        self._state = 'build_training_data'
        self._transformer = None
        self._batch_indices = []
        self._batch_indices_val = []
        self._current_batch = []
        self._current_batch_val = []
        self._batch_counter = 0
        self._batch_counter_val = 0
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Net = ImageNet(self._input_dim, self._hidden_layers, self._dropout, self._num_head, self._input_dim)
        self.Net.to(self._device)

    def forward(self, X, Image_X):
        return self.Net(X, Image_X)

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
            x = np.array(feature_list[2:])
            y = feature_list[1]
            pname = feature_list[0]
            if pname in self._training_name.keys():
                self._training_name[pname]['x'].append(x)
                self._training_name[pname]['y'].append(y)
                self._training_name[pname]['pname'].append(pname)
            else:
                if pname == '<NA>':
                    return
                self._training_name[pname] = {}
                self._training_name[pname]['x'] = [x]
                self._training_name[pname]['y'] = [y]
                self._training_name[pname]['pname'] = [pname]

            # self._training_data['x'].append(x)
            # self._training_data['y'].append(y)
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
            x = np.array(feature_list[2:])
            y = feature_list[1]
            pname = feature_list[0]
            # self._validation_name['pname'].append(pname)
            # self._validation_name[pname]['x'].append(x)
            # self._validation_name[pname]['y'].append(y)

            if pname in self._validation_name.keys():
                self._validation_name[pname]['x'].append(x)
                self._validation_name[pname]['y'].append(y)
                self._validation_name[pname]['pname'].append(pname)
            else:
                if pname == '<NA>':
                    return
                self._validation_name[pname] = {}
                self._validation_name[pname]['x'] = [x]
                self._validation_name[pname]['y'] = [y]
                self._validation_name[pname]['pname'] = [pname]
            # self._validation_data['x'].append(x)
            # self._validation_data['y'].append(y)
            self._current_batch_val.append(self._batch_counter_val)
            self._batch_counter_val += 1
            if len(self._current_batch_val) == self._batch_size:
                self._batch_indices_val.append(self._current_batch_val)
                self._current_batch_val = []
        else:
            pass

    def _create_train_data(self, val: bool):
        if not val:
            _training_data = {'x': [], 'y': [], 'pname': []}
            for key in self._training_name.keys():
                if key== '<NA>':
                    continue
                _training_data['x'].extend(self._training_name[key]['x'])
                _training_data['y'].extend(self._training_name[key]['y'])
                _training_data['pname'].extend(self._training_name[key]['pname'])

            x_tensors = Variable(torch.Tensor(_training_data['x'])).to(self._device)
            y_tensors = Variable(torch.Tensor(_training_data['y'])).to(self._device)
            y_tensors = y_tensors.long()
            x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Training Shape x: {x_tensors_final.shape} y: {y_tensors.shape}")
            return SyscallFeatureDataSet(x_tensors_final, y_tensors, _training_data['pname']), y_tensors
        else:
            _validation_data = {'x': [], 'y': [], 'pname': []}
            for key in self._validation_name.keys():
                if key== '<NA>':
                    continue
                _validation_data['x'].extend(self._validation_name[key]['x'])
                _validation_data['y'].extend(self._validation_name[key]['y'])
                _validation_data['pname'].extend(self._validation_name[key]['pname'])

            x_tensors = Variable(torch.Tensor(_validation_data['x'])).to(self._device)
            y_tensors = Variable(torch.Tensor(_validation_data['y'])).to(self._device)
            y_tensors = y_tensors.long()
            x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Validation Shape x: {x_tensors_final.shape} y: {y_tensors.shape}")
            return SyscallFeatureDataSet(x_tensors_final, y_tensors, _validation_data['pname']), y_tensors

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

        def xavier_init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == nn.GRU:
                for param in m._flat_weights_names:
                    if "weight" in param:
                        nn.init.xavier_uniform_(m._parameters[param])

        self.Image = {}
        f = open('K:\hids\dataSet\Image\\apache2', "rb")
        hex_list = [c for c in f.read()]
        f.close()
        self.Image['apache2'] = torch.stack(torch.Tensor(hex_list).split(64)[:-1]).unsqueeze(0).to(self._device)

        f = open('K:\hids\dataSet\Image\\mysql', "rb")
        hex_list = [c for c in f.read()]
        f.close()
        self.Image['mysqld'] = torch.stack(torch.Tensor(hex_list).split(64)[:-1]).unsqueeze(0).to(self._device)

        if self._state == 'build_training_data':
            self._state = 'fitting'
        if self.Net is not None:
            train_dataset, y_tensors = self._create_train_data(val=False)
            val_dataset, y_tensors_val = self._create_train_data(val=True)
            # for custom batches
            train_dataloader = DataLoader(train_dataset, batch_sampler=self._batch_indices)
            val_dataloader = DataLoader(val_dataset, batch_sampler=self._batch_indices_val)

            # self.Net.apply(xavier_init_weights)
            preds = []
            # Net hyperparameters
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.Net.parameters(), lr=0.001)
            torch.manual_seed(1)
            for epoch in tqdm(range(self._epochs),
                              'training network:'.rjust(25),
                              unit=" epochs"):
                for i, data in enumerate(train_dataloader, 0):
                    inputs, labels, proname = data
                    # def forward(self, X, Image_X)
                    outputs = self.Net.forward(inputs, self.Image[proname[0]])
                    optimizer.zero_grad()
                    # obtain the loss function
                    train_loss = criterion(outputs, labels)
                    # calculates the loss of the loss function
                    train_loss.backward()
                    # improve from loss, i.e backpro, val_loss: %1.5fp
                    optimizer.step()
                    for j in range(len(outputs)):
                        preds.append(torch.argmax(outputs[j]))
                accuracy = self._accuracy(preds, y_tensors)
                preds = []
                # reset hidden state
                self.new_recording()
                val_loss = 0.0
                for data in val_dataloader:
                    inputs, labels, proname = data
                    outputs = self.Net.forward(inputs, self.Image[proname[0]])
                    optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    val_loss = loss.item() * inputs.size(0)
                    for j in range(len(outputs)):
                        preds.append(torch.argmax(outputs[j]))
                val_accuracy = self._accuracy(preds, y_tensors_val)
                preds = []
                print("Epoch: %d, loss: %1.5f, accuracy: %1.5f, val_loss: %1.5f,  val_accuracy: %1.5f" %
                      (epoch,
                       train_loss.item(),
                       accuracy,
                       val_loss,
                       val_accuracy))
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
            pname = feature_list[0]
            x_tensor = Variable(torch.Tensor(np.array([feature_list[2:]])))
            x_tensor_final = torch.reshape(x_tensor,
                                           (x_tensor.shape[0],
                                            1,
                                            x_tensor.shape[1])).to(self._device)
            actual_syscall = feature_list[1]
            prediction_logits  = self.Net(x_tensor_final, self.Image[pname])
            softmax = nn.Softmax(dim=0)
            predicted_prob = float(softmax(prediction_logits[0])[actual_syscall])
            anomaly_score = 1 - predicted_prob
            return anomaly_score
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
        return hit/(hit+miss)

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
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        # dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(enc_outputs).view(-1, 61)

class SyscallFeatureDataSet(Dataset):

    def __init__(self, X, Y, pname):
        self.X = X
        self.Y = Y
        self.name = pname
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match length of Y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y, self.name[index]