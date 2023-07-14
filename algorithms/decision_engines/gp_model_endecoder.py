
from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
from tqdm import tqdm
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np

ENCODING_DIM = DECODING_DIM = 2
BOTTLENECK = 1

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


class GP_Encoder_Decoder(BuildingBlock):
    def __init__(self, input_vector: BuildingBlock, input_dim: int, epochs=200, batch_size=50):
        super().__init__()

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
        self.model = TrainModel(input_dim)
        self.model.to(self._device)
    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall):
        feature_list = self._input_vector.get_result(syscall)
        if feature_list is not None:
            self._training_data.append(feature_list)
            self._current_batch.append(self._batch_counter)
            self._batch_counter += 1
            if len(self._current_batch) == self._batch_size:
                self._batch_indices.append(self._current_batch)
                self._current_batch = []
        else:
            pass

    def val_on(self, syscall: Syscall):
        feature_list = self._input_vector.get_result(syscall)
        if feature_list is not None:
            self._validation_data.append(feature_list)

            self._current_batch_val.append(self._batch_counter_val)
            self._batch_counter_val += 1
            if len(self._current_batch_val) == self._batch_size:
                self._batch_indices_val.append(self._current_batch_val)
                self._current_batch_val = []
        else:
            pass

    def _create_train_data(self, val: bool):
        if not val:
            x_tensors = Variable(torch.Tensor(self._training_data)).to(self._device)
            print(f"Training Shape x: {x_tensors.shape}")
            return SyscallFeatureDataSet(x_tensors)
        else:
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

            # Net hyperparameters
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
            criterion = nn.MSELoss()

            for epoch in tqdm(range(self._epochs),
                              'training network:'.rjust(25),
                              unit=" epochs"):
                self.model.train()
                for i, data in enumerate(train_dataloader, 0):
                    data = data.reshape(-1, 1)
                    outputs = self.model(data)
                    optimizer.zero_grad()
                    classify_loss = criterion(outputs, data)
                    # regularization_loss = 0
                    # for param in self.model.parameters():
                    #     regularization_loss += torch.sum(abs(param))
                    #
                    # loss = classify_loss + 0.01 * regularization_loss

                    classify_loss.backward()
                    optimizer.step()

                self.model.eval()
                val_loss = 0.0
                for data in val_dataloader:
                    data = data.reshape(-1, 1)
                    outputs = self.model(data)
                    optimizer.zero_grad()
                    val_loss = criterion(outputs, data)


                print("Epoch: %d, loss: %1.5f val loss %1.5f" % (epoch, classify_loss.item(), val_loss.item()))

    def _calculate(self, syscall: Syscall):
        feature_list = self._input_vector.get_result(syscall)
        if feature_list is not None:
            x_tensor = Variable(torch.Tensor([feature_list]))
            x_tensor_final = torch.reshape(x_tensor, (-1, 1)).to(self._device)

            prediction_logits  = self.model(x_tensor_final)
            training_reconstruction_errors = prediction_logits.cpu().detach()[0][0].item()
            print(f'USI {x_tensor_final} mode calculate result {training_reconstruction_errors}')
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