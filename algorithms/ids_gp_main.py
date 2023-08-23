import os
import sys

from pprint import pprint

from datetime import datetime

from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.time_delta import TimeDelta
from algorithms.features.impl.max_score_threshold import MaxScoreThreshold

from algorithms.ids import IDS
from algorithms.persistance import save_to_mongo

from dataloader.direction import Direction

from algorithms.decision_engines.scg import SystemCallGraph
from algorithms.features.impl.return_value import ReturnValue
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.w2v_embedding import W2VEmbedding
from algorithms.features.impl.concat import Concat
from algorithms.features.impl.usi import Usi
from algorithms.features.impl.tid_number import TidNumber
from algorithms.features.impl.sequence_period import Sequence_period
from algorithms.features.impl.seen_syscall_list import SeenSysC
from algorithms.features.impl.scg_sc_freq import Syscall_Frequency
from algorithms.features.impl.gp_scg_period import Scg_Seq

from dataloader.data_loader_2021_df import DF_DataLoader2021
from dataloader.dataloader_factory import dataloader_factory

from algorithms.decision_engines.gp_model_endecoder import GP_Encoder_Decoder

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

    # feature config:
    NGRAM_LENGTH = 7
    W2V_SIZE = 4
    THREAD_AWARE = True

    # getting the LID-DS base path from argument or environment variable
    LID_DS_BASE_PATH = 'K:/hids'

    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(LID_DS_BASE_PATH,
                                     "dataSet",
                                     scenario_name)

        dataloader = DF_DataLoader2021(scenario_path, direction=Direction.BOTH)

        seq_period = Sequence_period()
        ssc = SeenSysC(seq_period)
        usi = Usi(seq_period, ssc)
        # ss = Scg_Seq(seq_period)
        # # tn = TidNumber(seq_period)
        # sf = Syscall_Frequency(ss)
        # features = [usi, sf]
        # concat = Concat(features)

        ed = GP_Encoder_Decoder(seq_period, 43)

        # features
        # 'time','UserID', 'PID', 'ProcessName', 'TID', 'syscall', 'DIR', 'ARGS'
        # 1. 每秒 时间间隔 数量， 针对每一项进行频率统计
        # 2. 每秒 TID 的频率统计
        # 3. syscall call graph create, user intEmbedding + return value(default 0)
        # TODO: PID and TID change fraquency
        # TODO: ARGs Analysis
        ###################

        # syscallName = SyscallName()
        # intEmbedding = IntEmbedding(syscallName)
        #
        # td = TimeDelta(thread_aware=True)
        # rv = ReturnValue()
        # ngram = Ngram([intEmbedding, rv], THREAD_AWARE, NGRAM_LENGTH)
        #
        # scg = SystemCallGraph(ngram)
        #
        decider = MaxScoreThreshold(ed)

        ###################
        # the IDS
        ids = IDS(data_loader=dataloader,
                  resulting_building_block=decider,
                  create_alarms=False,
                  plot_switch=False)

        ids.determine_threshold()
        # detection
        performance = ids.detect_batchs()
        #
        results = performance.get_results()
        #
        pprint(results)

        # enrich results with configuration and save to mongoDB
        results['config'] = ids.get_config_tree_links()
        results['scenario'] = scenario_name
        results['ngram_length'] = NGRAM_LENGTH
        results['thread_aware'] = THREAD_AWARE
        results['w2v_size'] = W2V_SIZE
        results['dataset'] = LID_DS_VERSION[LID_DS_VERSION_NUMBER]
        results['direction'] = dataloader.get_direction_string()
        results['date'] = str(datetime.now().date())
        results['model'] = "IDS-SCG"
        #
        # save_to_mongo(results)

        print('End')
