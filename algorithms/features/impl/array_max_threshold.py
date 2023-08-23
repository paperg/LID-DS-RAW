"""
Building Block for max value of training threshold.
"""
import numpy as np
import pandas as pd

from dataloader.syscall import Syscall
from algorithms.building_block import BuildingBlock


class ArrayMaxThreshold(BuildingBlock):
    """
        Saves maximum anomaly score of validation data as threshold.
    """

    def __init__(self,
                feature: BuildingBlock,
                use_timedelta = True,
                use_ptidfreq = True,
                use_usa = True,
                use_usc = True,
                use_freq = True):

        super().__init__()
        self._threshold = 0

        self._feature = feature
        self._dependency_list = []
        self._dependency_list.append(self._feature)

        self._use_timedelta = use_timedelta
        self._use_ptidfreq = use_ptidfreq
        self._use_usa = use_usa
        self._use_usc = use_usc
        self._use_freq = use_freq

        self.max_array = np.zeros(use_timedelta + use_ptidfreq * 4 + use_usa + use_usc + use_freq)
    def depends_on(self):
        return self._dependency_list

    def calculate_result(self, res_array, val=True):
        index = 0
        d = []
        if self._use_timedelta:
            d = res_array[:, :39].mean(axis=1).reshape(-1, 1)
            index += 39

        if self._use_ptidfreq:
            if len(d) > 0:
                d = np.concatenate((d, res_array[:, index:index + 4]), axis=1)
            else:
                d = res_array[:, index:index + 4]

            index += 4

        if self._use_usa:
            if len(d) > 0:
                d = np.concatenate((d, res_array[:, index:index + 1]), axis=1)
            else:
                d = res_array[:, index:index + 1]
            index += 1

        if self._use_usc:
            if len(d) > 0:
                d = np.concatenate((d, res_array[:, index:index + 1]), axis=1)
            else:
                d = res_array[:, index:index + 1]
            index += 1

        if self._use_freq:
            if len(d) > 0:
                d = np.concatenate((d, res_array[:, index:index + 1]), axis=1)
            else:
                d = res_array[:, index:index + 1]
            index += 1

        if val:
            d = d.max(axis=0)

        return d
    def val_on(self, syscall):
        """
        save highest seen anomaly_score
        """
        if syscall[0] is not None:
            res_array = self._feature.get_result(syscall)
            if isinstance(res_array, np.ndarray):
                d = self.calculate_result(res_array)
                self.max_array = np.fmax(self.max_array, d)

    def fit(self):
        self.max_array = self.max_array * 2
        print(f'Max Score {self.max_array}')
    def _calculate(self, syscall: Syscall) -> bool:
        """
        Return 0 if anomaly_score is below threshold.
        Otherwise return 1.
        """
        res_array = self._feature.get_result(syscall)
        if isinstance(res_array, np.ndarray):
            d = np.mean(res_array[:39], axis=0)
            d = np.append(d, res_array[39:])
            arr = np.greater(self.max_array, d)
            if np.all(arr):
                return False
            else:
                return True
        return False

    def get_batch_result(self):
        result, timestaplist = self._feature.cal_test_result()
        self._feature.test_batch_finish()
        if result is not None:
            anomaly_result = []
            arr_list = []
            score_result = self.calculate_result(result, False)
            for score in score_result:
                arr = np.greater_equal(self.max_array, score)
                arr_list.append(arr)
                if np.all(arr):
                    anomaly_result.append(0)
                else:
                    anomaly_result.append(1)

            return anomaly_result, timestaplist, result, arr_list
        else:
            print('There no Result return in get_batch_result() function,Please Check!')
            return None

    def is_decider(self):
        return True
