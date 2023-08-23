"""
    IDS class definition
"""
import pandas as pd
import numpy as np
import os

from copy import deepcopy
from functools import reduce
from typing import Type

from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from algorithms.building_block import BuildingBlock
from algorithms.data_preprocessor import DataPreprocessor, GP_DataPreprocessor
from algorithms.performance_measurement import Performance
from algorithms.score_plot import ScorePlot
from algorithms.util.dependency_graph_encoding import dependency_graph_to_config_tree
from dataloader.base_data_loader import BaseDataLoader
from dataloader.base_recording import BaseRecording


class IDS:
    """
        Intrusion Detection System Class
        Combines data loading, data processing and performance analysis
        Final BuildingBlock needs to be a decider which returns 0 if no anomaly has been detected.
    """
    def __init__(self,
                 data_loader: BaseDataLoader,
                 resulting_building_block: BuildingBlock,
                 plot_switch: bool,
                 create_alarms: bool = False):
        self._data_loader = data_loader
        self._final_bb = resulting_building_block
        if not self._final_bb.is_decider():
            raise ValueError('Resulting BuildingBlock is not a decider!')
        self._data_preprocessor = GP_DataPreprocessor(self._data_loader, resulting_building_block)
        self.threshold = 0.0
        self._alarm = False
        self._anomaly_scores_exploits = []
        self._anomaly_scores_no_exploits = []
        self._first_syscall_after_exploit_list = []
        self._last_syscall_of_recording_list = []
        self._create_alarms = create_alarms
        self.performance = Performance(create_alarms)
        if plot_switch is True:
            self.plot = ScorePlot(data_loader.scenario_path)
        else:
            self.plot = None

    def get_config(self) -> str:
        return self._data_preprocessor.get_graph_dot()

    def get_config_tree_links(self) -> dict:
        """
            gives the dependency graph as list of links between ids of building blocks
            each building block contains its config in another list called node

            returns: dictionary with nodes and links of the config graph
        """

        return dependency_graph_to_config_tree(
            self._data_preprocessor.get_building_block_manager().get_dependency_graph()
        )

    def determine_threshold(self):
        """
        decision engine calculates anomaly scores using validation data,
        saves biggest score as threshold for detection phase

        """
        max_score = 0.0
        data = self._data_loader.validation_data()
        description = 'Threshold calculation'.rjust(27)
        for recording in tqdm(data, description, unit=" recording"):
            for syscall in recording.syscalls():
                anomaly_score = self._final_bb.get_result(syscall)
                if anomaly_score is not None:
                    if anomaly_score > max_score:
                        max_score = anomaly_score
            self._data_preprocessor.new_recording()
        self.threshold = max_score
        self.performance.set_threshold(max_score)
        if self.plot is not None:
            self.plot.threshold = max_score
        print(f"threshold={max_score:.3f}".rjust(27))

    def determine_threshold_and_plot(self):
        """
        decision engine calculates anomaly scores using validation data,
        saves biggest score as threshold for detection phase
        plots the validation scores
        """
        max_score = 0.0
        data = self._data_loader.validation_data()
        description = 'Threshold calculation'.rjust(27)
        scores = []
        for recording in tqdm(data, description, unit=" recording"):
            for syscall in recording.syscalls():
                anomaly_score = self._final_bb.get_result(syscall)
                if anomaly_score is not None:
                    scores.append(anomaly_score)
                    if anomaly_score > max_score:
                        max_score = anomaly_score
            self._data_preprocessor.new_recording()
        self.threshold = max_score
        self.performance.set_threshold(max_score)
        if self.plot is not None:
            self.plot.threshold = max_score
        print(f"threshold={max_score:.3f}".rjust(27))

        plt.plot(scores)
        plt.show()

    def detect(self) -> Performance:
        """
        detecting performance values using the test data,
        calling performance object for measurement and
        plot object if plot_switch is True
        """
        data = self._data_loader.test_data()
        description = 'anomaly detection'.rjust(27)

        for recording in tqdm(data, description, unit=" recording"):
            self.performance.new_recording(recording)
            if self.plot is not None:
                self.plot.new_recording(recording)

            for syscall in recording.syscalls():
                is_anomaly = self._final_bb.get_result(syscall)
                self.performance.analyze_syscall(syscall, is_anomaly)
                if self.plot is not None:
                    self.plot.add_to_plot_data(anomaly_score,
                                               syscall,
                                               self.performance.get_cfp_indices())

            self._data_preprocessor.new_recording()

            # run end alarm once to ensure that last alarm gets saved
            if self.performance.alarms is not None:
                self.performance.alarms.end_alarm()
        return self.performance

    def detect_batchs(self, model) -> Performance:
        """
        detecting performance values using the test data,
        calling performance object for measurement and
        plot object if plot_switch is True
        """
        data = self._data_loader.test_data()
        description = 'anomaly detection'.rjust(27)
        analyze_dir = 'K:\\hids\\dataAnalyze'
        for recording in tqdm(data, description, unit=" recording"):
            self.performance.new_recording(recording)
            if self.plot is not None:
                self.plot.new_recording(recording)

            columns = ['file_exploit_time', 'current_time']
            if model._use_timedelta:
                columns.extend(['Time_result'])

            if model._use_ptidfreq:
                columns.extend(['PID_SW_Freq', 'PID_Number', 'TID_SW_Freq', 'TID_Number'])

            if model._use_usa:
                columns.extend(['UnseenARGS'])

            if model._use_usc:
                columns.extend(['UnseenSC'])

            if model._use_freq:
                columns.extend(['Freq'])

            df = pd.DataFrame(columns = columns + [i for i in range(model._input_dim)])
            nor_df = pd.DataFrame(columns = columns + [i for i in range(model._input_dim)])

            for data_tuple in recording.df_and_np():
                if model.get_input_result(data_tuple):
                    is_anomaly, timestaps, result_data, result_bools = self._final_bb.get_batch_result()
                    for index, cur_time in enumerate(timestaps):
                        need_hanle, current_exploit_time = self.performance.analyze_batchs(cur_time, is_anomaly[index])
                        if need_hanle:
                            new_row = np.array([current_exploit_time, cur_time])
                            new_row = np.append(new_row, result_bools[index])
                            new_row = np.append(new_row, result_data[index])
                            df.loc[len(df.index)] = new_row
                            # print(result_data[i])
                        elif current_exploit_time is not None:
                            new_row = np.array([current_exploit_time, cur_time])
                            new_row = np.append(new_row, result_bools[index])
                            new_row = np.append(new_row, result_data[index])
                            nor_df.loc[len(nor_df.index)] = new_row

                    if self.plot is not None:
                        self.plot.add_to_plot_data(anomaly_score,
                                                   syscall,
                                                   self.performance.get_cfp_indices())
            if current_exploit_time is not None:
                if self.performance._alarm is False:
                    print(f'File {recording.name} Not recorgnize')

            self._data_preprocessor.new_recording()
            if len(df) > 0:
                df.to_pickle(os.path.join(analyze_dir, recording.name + '_exploit.pkl'))
            if len(nor_df) > 0:
                nor_df.to_pickle(os.path.join(analyze_dir, recording.name + '_normal.pkl') )

            # run end alarm once to ensure that last alarm gets saved
            if self.performance.alarms is not None:
                self.performance.alarms.end_alarm()

        return self.performance

    def detect_on_single_recording(self, recording: Type[BaseRecording]) -> Performance:
        """
        detecting performance values using single recording
        create Performance object and return it

        Args:
            recording: single recording to calculate performance on
        Returns:
            Performance: performance object
        """
        performance = Performance(self._create_alarms)
        performance.set_threshold(self.threshold)

        # Wenn das eine Exploit-Aufnahme ist, dann schreibe den Zeit-Stempel auf
        if recording.metadata()["exploit"]:
            performance.set_exploit_time(recording.metadata()["time"]["exploit"][0]["absolute"])
            performance._exploit_count += 1

        for syscall in recording.syscalls():
            is_anomaly = self._final_bb.get_result(syscall)
            performance.analyze_syscall(syscall, is_anomaly)

        self._data_preprocessor.new_recording()

        # End alarms because end of recording is reached
        performance._cfp_end_normal()
        performance._cfp_end_exploits()

        # run end alarm once to ensure that last alarm gets saved
        if performance.alarms is not None:
            performance.alarms.end_alarm()

        return performance

    def draw_plot(self, filename=None):
        # plot data if wanted
        if self.plot is not None:
            self.plot.feed_figure()
            self.plot.show_plot(filename)

    def _calculate(recording_ids_tuple: tuple) -> Performance:
        """
            create deepcopy of IDS and get performance object for recording of container

            Args:
            recroding_ids_tuple:
                ids: IDS with which perfomance is calculated
                recording: Recording on which performance is calculated
        """
        # get ids (as deep copy) and recording
        ids = deepcopy(recording_ids_tuple[0])
        recording = recording_ids_tuple[1]
        # Calculate performance on current recording and return it
        return ids.detect_on_single_recording(recording)

    def detect_parallel(self) -> Performance:
        """
            map reduce for every recording
            map:    first calculate performances on each single recording with ids
            reduce: than sum up performances

            Returns:
                Performance: complete performance of all recordings

        """
        # creating list of Tuples with deepcopys of this ids object and recordings
        ids_and_recordings = [(self, recording) for recording in self._data_loader.test_data()]

        # parallel calculation for every recording
        performance_list = process_map(
            IDS._calculate,
            ids_and_recordings,
            chunksize=20,
            desc="anomaly detection".rjust(27),
            unit=" recordings")

        # Sum up performances
        if self._create_alarms:
            final_performance = reduce(Performance.add_with_alarms, performance_list)
        else:
            final_performance = reduce(Performance.add, performance_list)

        return final_performance
