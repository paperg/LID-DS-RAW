'''
    example script for IDS on cluster
'''
import time
import math
import logging
import argparse
import traceback
import sys
import os
sys.path.append('.')
from pprint import pprint

from dataloader.direction import Direction
from dataloader.dataloader_factory import dataloader_factory
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.max_score_threshold import MaxScoreThreshold
from algorithms.ids import IDS

# from algorithms.features.impl.mode import Mode
# from algorithms.features.impl.flags import Flags
from algorithms.features.impl.ngram import Ngram
# from algorithms.features.impl.concat import Concat
# from algorithms.features.impl.process_name import ProcessName
from algorithms.features.impl.int_embedding import IntEmbedding

from algorithms.decision_engines.stide import Stide
from dataloader.dataset_create_gp import DATAOUT_DIR, work_dir
from algorithms.persistance import save_to_mongo

if __name__ == '__main__':

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


    thread_aware = True
    window_length = 64
    ngram_length = 6
    embedding_size = 3
    hidden_size = int(math.sqrt(ngram_length * embedding_size))
    direction = Direction.BOTH
    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(work_dir,
                                     scenario_name)
        dataloader = dataloader_factory(scenario_path, direction=direction)
        syscall_embedding = IntEmbedding(scenario_name=scenario_name)
        ngram = Ngram([syscall_embedding], thread_aware, ngram_length)
        # finally calculate the STIDE algorithm using these ngrams
        de = Stide(ngram)

        stream_sum = StreamSum(de, thread_aware, window_length, False)
        decider = MaxScoreThreshold(stream_sum)
        ### the IDS
        ids = IDS(data_loader=dataloader,
                  resulting_building_block=decider,
                  create_alarms=True,
                  plot_switch=False)

        print("at evaluation:")
        # threshold
        ids.determine_threshold()
        # detection
        start = time.time()
        performance = ids.detect_parallel()
        end = time.time()

        detection_time = (end - start)/60  # in min

        print(detection_time)
        ### print results and plot the anomaly scores
        results = performance.get_results()
        pprint(results)
        if direction == Direction.BOTH:
            DIRECTION = 'BOTH'
        elif direction == Direction.OPEN:
            DIRECTION = 'OPEN'
        else:
            DIRECTION = 'CLOSE'
        results['dataset'] = 'LID-DS-2021'
        results['scenario'] = scenario
        results['ngram_length'] = ngram_length
        results['embedding'] = 'INT'
        results['algorithm'] = 'STIDE_BASE'
        results['stream_sum'] = window_length
        results['detection_time'] = detection_time
        results['config'] = ids.get_config()
        results['thread_aware'] = thread_aware
        results['flag'] = False
        results['mode'] = False
        results['cluster'] = True
        results['parallel'] = False
        results['process_name'] = False
        # save_to_mongo(results)