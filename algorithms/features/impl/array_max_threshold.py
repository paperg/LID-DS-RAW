"""
Building Block for max value of training threshold.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader.syscall import Syscall
from algorithms.building_block import BuildingBlock


class ArrayMaxThreshold(BuildingBlock):
    """
        Saves maximum anomaly score of validation data as threshold.
    """

    def __init__(self,
                feature: BuildingBlock,
                use_timedelta = False,
                use_ret = False,
                use_usa = False,
                use_usc = False,
                use_freq = False,
                use_sc_max_params = False,
                use_my_mode = True):

        super().__init__()
        self._threshold = 0

        self._feature = feature
        self._dependency_list = []
        self._dependency_list.append(self._feature)

        self._use_timedelta = use_timedelta
        self._use_ret = use_ret
        self._use_usa = use_usa
        self._use_usc = use_usc
        self._use_freq = use_freq
        self._use_sc_max_params = use_sc_max_params
        self.use_my_model = use_my_mode
        if use_my_mode:
            self.max_array = np.zeros(use_timedelta + use_ret + use_usa + use_usc + use_freq + use_sc_max_params * 12)
        else:
            self.max_array = 0
        self.loss_pic_index = 0
    def depends_on(self):
        return self._dependency_list

    # 计算结果，返回结果数组
    def calculate_result(self, res_array, val=True):
        index = 0
        d = []
        if self._use_timedelta:
            d = res_array[:, :39].mean(axis=1).reshape(-1, 1)
            index += 39

        # if self._use_ptidfreq:
        #     if len(d) > 0:
        #         d = np.concatenate((d, res_array[:, index:index + 4]), axis=1)
        #     else:
        #         d = res_array[:, index:index + 4]
        #
        #     index += 4

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

        if self._use_ret:
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

        if self._use_sc_max_params:
            # res = res_array[:, index:index+8].mean(axis=1).reshape(-1, 1)
            res = res_array[:, index:index + 12]
            if len(d) > 0:
                d = np.concatenate((d, res), axis=1)
            else:
                d = res

        if val:
            d = d.max(axis=0)

        return d
    def val_on(self, syscall):
        """
        save highest seen anomaly_score
        """
        if syscall[0] is not None:
            res_array = self._feature.get_result(syscall)
            if self.use_my_model:
                if isinstance(res_array, np.ndarray):
                    d = self.calculate_result(res_array)
                    self.max_array = np.fmax(self.max_array, d)
            else:
                if res_array is not None:
                    self.max_array = max(res_array, self.max_array)

    def fit(self):
        print(f'Max Score {self.max_array}')
    def _calculate(self, syscall: Syscall) -> bool:
        print('Call Error, Check !')
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

    # result is anomaly result, last is last time_score
    def  gp_func(self, result, last_time_score):
        # Delta = np.array([15.0] * len(self.max_array))
        # # for usa
        # # Delta[0] = Delta[0] * 0.5
        # # # for syscall frequency
        # # # 对于调用频率，输入在 0 - 1， 偏差会小
        # # Delta[2:] = Delta[2:] * 0.5
        #
        # alpha = 0.4
        # # res 随之 1 减小， 0 增大
        # time_score = (1 / np.exp((1 - result) * alpha) + (result) * alpha) * last_time_score
        # # 限制最小值
        # time_score = np.fmax(time_score, 1/15)
        # # 限制最大值
        # time_score = np.fmin(time_score, 1)
        #
        # res = np.multiply(Delta, time_score)
        # # 最终结果不能大于 1， 要使用 max score * res, max_score 已经是判断异常的最低标准
        # res = np.fmax(res, 1)
        #
        # return res, time_score
        b = 1
        c = 0.3
        x = (1 / (1 + np.exp(-b * (result - last_time_score))))
        # y = 1 - np.exp(-c * (1 - (result * (result - last_time_score) - 0.1) / 0.9))
        w = (1 - (result * (result - last_time_score)))
        z = 1 + np.exp(-50 * (last_time_score - 0.2))
        y = 1 - np.exp(-c * w * z)

        return (x * y)

    def get_batch_result(self):
        result, rec_loss1, inputs, timestaplist = self._feature.cal_test_result()

        if self.use_my_model:
            time_score = np.array([1] * len(self.max_array))
            last_result = np.array([1.0] * len(self.max_array))
            time_result = np.array([1] * len(self.max_array))
            Delta = np.array([15.0] * len(self.max_array))
            if result is not None:
                anomaly_result = []
                arr_list = []
                max_score_list = []
                score_result = self.calculate_result(result, False)
                need_show = False
                # six_dim = False
                for index, score in enumerate(score_result):

                    # time_score = self.gp_func(last_result, time_result)
                    #
                    # time_result = time_result * (time_score + last_result)
                    # time_result = np.fmin(time_result, 1)
                    # time_result = np.fmax(time_result, 0.06)
                    #
                    # comp_array = self.max_array * np.multiply(Delta, time_result)
                    comp_array = self.max_array * 10
                    arr = np.greater_equal(comp_array, score)
                    # last_result = arr
                    max_score_list.append(comp_array)
                    arr_list.append(arr)
                    # if np.sum[arr == False]) > 1:
                    #
                    # else:
                    if np.all(arr):
                        last_result[:] = 1
                        anomaly_result.append(0)
                    else:
                        last_result[:] = 0
                        anomaly_result.append(1)

                find_one = False
                for i in range(len(anomaly_result)):
                    if anomaly_result[i] == 0:
                        if find_one:
                            need_show = True
                    else:
                        find_one = True

                if need_show:
                    if self.loss_pic_index < 20:
                        max_score_list = np.array(max_score_list)
                        self.loss_display(rec_loss1, result, arr_list, max_score_list, save_path=f"K:/hids/dataAnalyze/loss_analyze_{self.loss_pic_index}.svg")
                        self.loss_pic_index += 1

                return anomaly_result, inputs, timestaplist, result, arr_list
            else:
                print('There no Result return in get_batch_result() function,Please Check!')
                return None
        else:
            time_score = np.array([1])
            last_result = np.array([1.0])
            time_result = np.array([1])
            Delta = np.array([15.0])
            if result is not None:
                time_score = self.gp_func(last_result, time_result)

                time_result = time_result * (time_score + last_result)
                time_result = np.fmin(time_result, 1)
                time_result = np.fmax(time_result, 0.06)

                comp_array = self.max_array * np.multiply(Delta, time_result)
                anomaly_result = result > comp_array
                return anomaly_result, inputs, timestaplist, result, []
            else:
                return None

    def loss_display(self, loss1, loss2, arr_list, max_score_list, save_path = "./loss_analyze.svg"):
        dim = len(loss1[0])
        x_axis_data = [i for i in range(len(loss1))]

        figure = plt.figure(figsize=(75, 150))

        fig_list = []
        for i in range(dim):
            ax = figure.add_subplot(dim, 1, i + 1)
            fig_list.append(ax)

        for i, ax in enumerate(fig_list):
            ax.set_title(f"Rec Loss Dim {i + 1}", fontsize=40)
            ax.set_xlabel('Data Index', fontsize=40)
            ax.set_ylabel('Loss', fontsize=40)

            ax.plot(x_axis_data, loss1[:, i], 'g*--', alpha=0.7, linewidth=4, label='Rec Loss 1')
            ax.plot(x_axis_data, loss2[:, i], 'b^-.', alpha=0.7, linewidth=4, label='Rec Loss 2')
            ax.plot(x_axis_data, max_score_list[:, i], 'ro--', alpha=0.7, linewidth=5, label='Max Score')
            ax.legend(loc='upper right', fontsize=40)

            y_max = max(max(loss1[:, i]), max(loss2[:, i]))
            for x_index in range(len(arr_list)):
                if arr_list[x_index][i] == 0:
                    rect = plt.Rectangle((x_index - 0.3, 0.0), 0.6, y_max + 0.1, edgecolor='r', linewidth=5, facecolor='none')
                    ax.add_patch(rect)

        plt.savefig(save_path, dpi=400, format="svg")

    def is_decider(self):
        return True
