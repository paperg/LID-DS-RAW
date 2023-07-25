"""
example script for running LSTM
"""
import os
import sys
import time
import pandas as pd
import numpy as np

from pprint import pprint
from datetime import datetime
from tqdm import tqdm
from dataloader.direction import Direction
from dataloader.dataloader_factory import dataloader_factory

from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.concat import Concat
from algorithms.features.impl.time_delta import TimeDelta
from algorithms.features.impl.return_value import ReturnValue
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.features.impl.ngram_minus_one import NgramMinusOne
from algorithms.features.impl.thread_change_flag import ThreadChangeFlag
from algorithms.features.impl.SPOT import SpotThreshold
from algorithms.features.impl.SPOT import pot_eval
from algorithms.features.impl.process_name import ProcessName
from algorithms.features.impl.pname_ret import ProcessNameAndRet
# from algorithms.decision_engines.scg_gp import SystemCallGraph
# from algorithms.persistance import save_to_mongo

from algorithms.ids import IDS
from algorithms.decision_engines.tranAD import Transformer_ad

if __name__ == '__main__':

    LID_DS_VERSION_NUMBER = 1
    LID_DS_VERSION = [
            "LID-DS-2019",
            "LID-DS-2021"
            ]

    # scenarios ordered by training data size asc
    SCENARIOS = [
      "CWE-89-SQL-injection",
      "CVE-2017-7529",
      "CVE-2014-0160",
      "CVE-2012-2122",
      "Bruteforce_CWE-307",
      "CVE-2020-23839",
      "PHP_CWE-434",
      "ZipSlip",
      "CVE-2018-3760",
      "CVE-2020-9484",
      "EPS_CWE-434",
      "CVE-2019-5418",
      "Juice-Shop",
      "CVE-2020-13942",
      "CVE-2017-12635_6"
    ]
    SCENARIO_RANGE = SCENARIOS[0:1]

    # config
    # Transformer
    EPOCH=2
    DROPOUT=0.1
    HIDDEN_LAYERS = 1
    NUM_HEAD = 1

    NGRAM_LENGTH = 10
    EMBEDDING_SIZE = 4
    THREAD_AWARE = True
    BATCH_SIZE = 1024
    USE_THREAD_CHANGE_FLAG = True
    USE_RETURN_VALUE = True
    USE_TIME_DELTA = True


    # getting the LID-DS base path from argument or environment variable
    LID_DS_BASE_PATH = 'K:/hids'

    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(LID_DS_BASE_PATH,
                                     "dataSet",
                                     scenario_name)

        # data loader for scenario
        dataloader = dataloader_factory(scenario_path, direction=Direction.CLOSE)
        element_size = 1 + USE_RETURN_VALUE + USE_TIME_DELTA
        # embedding
        int_embedding = IntEmbedding()
        # w2v = W2VEmbedding(word=int_embedding,
        #                    vector_size=EMBEDDING_SIZE,
        #                    window_size=NGRAM_LENGTH,
        #                    epochs=5000)
        feature_list = [int_embedding]
        if USE_RETURN_VALUE:
            rv = ReturnValue()
            feature_list.append(rv)
        if USE_TIME_DELTA:
            td = TimeDelta(thread_aware=True)
            feature_list.append(td)
        pnr = ProcessNameAndRet(int_embedding)
        # sc = SystemCallGraph(pnr)
        # feature_list.append(sc)

        ngram = Ngram(
            feature_list=feature_list,
            thread_aware=THREAD_AWARE,
            ngram_length=NGRAM_LENGTH
        )
        # ngram_minus_one = NgramMinusOne(
        #     ngram=ngram,
        #     element_size=element_size
        # )

        # final_features = [int_embedding, ngram_minus_one]
        # if USE_THREAD_CHANGE_FLAG:
        #     tcf = ThreadChangeFlag(ngram_minus_one)
        #     final_features.append(tcf)
        # concat = Concat(final_features)

        model_path = f'Models/{scenario_name}/Transformer/'\
            f'drop{DROPOUT}' \
            f'hidLay{HIDDEN_LAYERS}' \
            f'NumH{NUM_HEAD}' \
            f'ta{THREAD_AWARE}' \
            f'ng{NGRAM_LENGTH}' \
            f'-emb{EMBEDDING_SIZE}' \
            f'-rv{USE_RETURN_VALUE}' \
            f'-td{USE_TIME_DELTA}' \
            f'-tcf{USE_THREAD_CHANGE_FLAG}.model'
        input_dim = (NGRAM_LENGTH * element_size +
                     USE_THREAD_CHANGE_FLAG)

        # decision engine (DE)
        distinct_syscalls = dataloader.distinct_syscalls_training_data()

        transAD = Transformer_ad(
                    input_vector=ngram,
                    distinct_syscalls=distinct_syscalls,
                    input_dim=element_size,
                    epochs=EPOCH,
                    dropout=DROPOUT,
                    num_head=NUM_HEAD,
                    hidden_layers=HIDDEN_LAYERS,
                    batch_size=BATCH_SIZE,
                    model_path=model_path,
                    force_train=False)

        decider = SpotThreshold(transAD)
        # define the used features
        print(type(transAD))
        ids = IDS(data_loader=dataloader,
                  resulting_building_block=decider,
                  create_alarms=False,
                  plot_switch=False)


        data = dataloader.test_data()
        description = 'anomaly detection'.rjust(27)
        labels = []
        current_exploit_time = None
        for recording in tqdm(data, description, unit=" recording"):
            if recording.metadata()["exploit"] is True:
                current_exploit_time = recording.metadata()["time"]["exploit"][0]["absolute"]
            else:
                current_exploit_time = None

            for syscall in recording.syscalls():
                ret = transAD.test_on(syscall)
                if ret is not None:
                    syscall_time = syscall.timestamp_unix_in_ns() * (10 ** (-9))
                    # syscall_time = syscall.iloc[-1]['time'].timestamp()
                    # files with exploit
                    if current_exploit_time is not None:
                        if current_exploit_time > syscall_time:
                            labels.append(0)
                        else:
                            labels.append(1)
                    else:
                        labels.append(0)
            ngram.new_recording()

        loss_test = transAD.cal_test_loss()
        df = pd.DataFrame()
        for i in range(loss_test.shape[1]):
            lt, l, ls = transAD._train_loss[1:, i], loss_test[1:, i], labels[:loss_test.shape[0] - 1]
            pot_eval(lt, l, ls)
            # df = df.append(result, ignore_index=True)

        print(df)

        print('End Exit')
        # start = time.time()
        # # detection
        # performance = ids.detect()
        # results = performance.get_results()
        # end = time.time()
        # detection_time = (end - start) / 60  # in min
        # pprint(results)
        #
        # results = transAM.get_results(results)
        # results['config'] = ids.get_config_tree_links()
        # results['element_size'] = element_size
        # results['ngram_length'] = NGRAM_LENGTH
        # results['thread_aware'] = THREAD_AWARE
        #
        # results['dataset'] = LID_DS_VERSION[LID_DS_VERSION_NUMBER]
        # results['direction'] = dataloader.get_direction_string()
        # results['date'] = str(datetime.now().date())
        # results['scenario'] = scenario_name
        # results['detection_time'] = detection_time
        # results['Model'] = 'Transforer_SSG'
        # save_to_mongo(results)
