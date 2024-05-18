
from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tqdm import tqdm
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math

from algorithms.decision_engines.Mine_AD import TranAD

ENCODING_DIM = DECODING_DIM = 2
BOTTLENECK = 1

class TrainModel(nn.Module):
    def __init__(self, input_dim):
        super(TrainModel, self).__init__()
        # L1
        self.L1 = nn.Linear(input_dim, input_dim * ENCODING_DIM)
        self.L1_A = nn.Sigmoid()
        # L2
        self.L2 = nn.Linear(input_dim * ENCODING_DIM, input_dim * BOTTLENECK)
        self.L2_A = nn.Sigmoid()
        # L3
        self.L3 = nn.Linear(input_dim * BOTTLENECK, input_dim * BOTTLENECK)
        self.L3_A = nn.Sigmoid()
        # L4
        self.L4 = nn.Linear(input_dim * BOTTLENECK, input_dim *DECODING_DIM)
        self.L4_A = nn.Sigmoid()
        # out
        self.out = nn.Linear(input_dim * DECODING_DIM, input_dim)
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


class AENetwork(nn.Module):
    """
    the actual autoencoder as torch module
    """

    def __init__(self, input_size):
        super().__init__()
        self._input_size = input_size
        self._factor = 0.7
        first_hidden_layer_size = self._input_size  # int(self._input_size * 1.333)
        # Building an encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self._input_size, first_hidden_layer_size),
            torch.nn.Dropout(p=0.5),
            torch.nn.SELU(),

            torch.nn.Linear(first_hidden_layer_size, int(first_hidden_layer_size * pow(self._factor, 2))),
            torch.nn.Dropout(p=0.5),
            torch.nn.SELU(),

            torch.nn.Linear(int(first_hidden_layer_size * pow(self._factor, 2)),
                            int(first_hidden_layer_size * pow(self._factor, 3))),
            torch.nn.Dropout(p=0.5),
            torch.nn.SELU(),

            torch.nn.Linear(int(first_hidden_layer_size * pow(self._factor, 3)),
                            int(first_hidden_layer_size * pow(self._factor, 4))),
            torch.nn.Dropout(p=0.5),
            torch.nn.SELU()
        )

        # Building an decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(int(first_hidden_layer_size * pow(self._factor, 4)),
                            int(first_hidden_layer_size * pow(self._factor, 3))),
            torch.nn.Dropout(p=0.5),
            torch.nn.SELU(),

            torch.nn.Linear(int(first_hidden_layer_size * pow(self._factor, 3)),
                            int(first_hidden_layer_size * pow(self._factor, 2))),
            torch.nn.Dropout(p=0.5),
            torch.nn.SELU(),

            torch.nn.Linear(int(first_hidden_layer_size * pow(self._factor, 2)), first_hidden_layer_size),
            torch.nn.Dropout(p=0.5),
            torch.nn.SELU(),

            torch.nn.Linear(first_hidden_layer_size, self._input_size),
            torch.nn.Dropout(p=0.5),
            # torch.nn.Sigmoid()
        )

        for m in self.encoder:
            if isinstance(m, nn.Linear):
                fan_in = m.in_features
                nn.init.normal_(m.weight, 0, math.sqrt(1. / fan_in))
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                fan_in = m.in_features
                nn.init.normal_(m.weight, 0, math.sqrt(1. / fan_in))

    def max_norm(self, max_val=2, eps=1e-8):
        for name, param in self.named_parameters():
            if 'bias' not in name:
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, 0, max_val)
                param = param * (desired / (eps + norm))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        self.max_norm()
        return decoded


class GP_Encoder_Decoder(BuildingBlock):
    def __init__(self, input_vector: BuildingBlock, input_dim: int, epochs=200, batch_size=1024):
        super().__init__()

        self._input_dim = input_dim
        self._epochs = epochs
        self._batch_size = batch_size
        # input_vector is USI currently
        self._input_vector = input_vector
        self._dependency_list = [input_vector]

        self._training_data = []
        self._validation_data = []

        self._batch_indices = []
        self._batch_indices_val = []

        self._batch_counter = 0
        self._current_batch = []
        self._batch_counter_val = 0
        self._current_batch_val = []
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._set_model(input_dim)

    def _set_model(self, input_dim):
        # self.model = TrainModel(input_dim)
        self.model = AENetwork(input_dim)
        self.model = TranAD(input_dim, 0.001)
        self.model.to(self._device)
    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall):
        # feature_list = self._input_vector.get_result(syscall)
        if syscall is not None:
            self._training_data.append(syscall[0])
            self._current_batch.append(self._batch_counter)
            self._batch_counter += 1
            if len(self._current_batch) == self._batch_size:
                self._batch_indices.append(self._current_batch)
                self._current_batch = []
        else:
            pass

    def val_on(self, syscall: Syscall):
        # feature_list = self._input_vector.get_result(syscall)
        if syscall is not None:
            self._validation_data.append(syscall[0])

            self._current_batch_val.append(self._batch_counter_val)
            self._batch_counter_val += 1
            if len(self._current_batch_val) == self._batch_size:
                self._batch_indices_val.append(self._current_batch_val)
                self._current_batch_val = []
        else:
            pass

    def _create_train_data(self, val: bool):
        if not val:
            df = pd.DataFrame(self._training_data)
            for i in range(39, 43):
                df[i] = df[i] / (df[i].max() + 1)
            self._training_data = np.array(df)
            x_tensors = Variable(torch.Tensor(self._training_data)).to(self._device)
            print(f"Training Shape x: {x_tensors.shape}")
            return SyscallFeatureDataSet(x_tensors)
        else:
            if len(self._validation_data) > 0:
                df = pd.DataFrame(self._validation_data)
                for i in range(39, 43):
                    df[i] = df[i] / (df[i].max() + 1)
                self._validation_data = np.array(df)

            x_tensors = Variable(torch.Tensor(self._validation_data)).to(self._device)
            print(f"Validation Shape x: {x_tensors.shape} ")
            return SyscallFeatureDataSet(x_tensors)

    def fit(self):

        if self.model is not None:
            train_dataset = self._create_train_data(val=False)
            val_dataset = self._create_train_data(val=True)
            # for custom batches
            train_dataloader = DataLoader(train_dataset, batch_sampler=self._batch_indices)
            val_dataloader = DataLoader(val_dataset, batch_sampler=self._batch_indices_val)
            #
            # # Net hyperparameters
            # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
            # criterion = nn.MSELoss()
            #
            # for epoch in tqdm(range(self._epochs),
            #                   'training network:'.rjust(25),
            #                   unit=" epochs"):
            #     self.model.train()
            #     for i, data in enumerate(train_dataloader, 0):
            #         # data = data.reshape(-1, self._input_dim)
            #         outputs = self.model(data)
            #         optimizer.zero_grad()
            #         classify_loss = criterion(outputs, data)
            #         # regularization_loss = 0
            #         # for param in self.model.parameters():
            #         #     regularization_loss += torch.sum(abs(param))
            #         #
            #         # loss = classify_loss + 0.01 * regularization_loss
            #
            #         classify_loss.backward()
            #         optimizer.step()
            #
            #     self.model.eval()
            #     val_loss = 0.0
            #     for data in val_dataloader:
            #         data = data.reshape(-1, self._input_dim)
            #         outputs = self.model(data)
            #         optimizer.zero_grad()
            #         val_loss = criterion(outputs, data)
            #
            #
            #     # print("Epoch: %d, loss: %1.5f val loss %1.5f" % (epoch, classify_loss.item(), val_loss.item()))
            #     print("Epoch: %d, loss: %1.5f " % (epoch, classify_loss.item()))

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
            criterion = nn.MSELoss()

            for epoch in tqdm(range(self._epochs),
                              'training network:'.rjust(25),
                              unit=" epochs"):
                n = epoch + 1
                self.model.train()
                for i, inputs in enumerate(train_dataloader, 0):
                    local_bs = inputs.shape[0]
                    inputs = inputs.view(local_bs, -1, self._input_dim)
                    window = inputs.permute(1, 0, 2)

                    reconstruct, rec2_input, reconstruct_resc = self.model(window)
                    train_loss = (1 / n) * criterion(window, reconstruct) + (1 - 1 / n) * criterion(rec2_input,
                                                                                                      reconstruct_resc)

                    optimizer.zero_grad()
                    # calculates the loss of the loss function
                    train_loss.backward()
                    # improve from loss, i.e backpro, val_loss: %1.5fp
                    optimizer.step()
                scheduler.step()

                print("Epoch: %d, loss: %1.5f " % (epoch, train_loss.item()))
    def _calculate(self, syscall: Syscall):
        feature_list = self._input_vector.get_result(syscall)
        if feature_list is not None:
            x_tensor = Variable(torch.Tensor([feature_list]))
            x_tensor_final = torch.reshape(x_tensor, (-1, self._input_dim)).to(self._device)

            prediction_logits  = self.model(x_tensor_final)
            training_reconstruction_errors = prediction_logits.cpu().detach()[0][0].item()
            print(f'Input {x_tensor_final} mode calculate result {training_reconstruction_errors}')
            return training_reconstruction_errors
        else:
            return None
class SyscallFeatureDataSet(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]