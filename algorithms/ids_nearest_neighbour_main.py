"""
Example execution of LIDS Framework
"""
import os
import sys
from pprint import pprint

from algorithms.features.impl.nearest_neighbour import NearestNeighbour
from dataloader.dataloader_factory import dataloader_factory

from dataloader.direction import Direction

from algorithms.ids import IDS

from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.features.impl.ngram import Ngram
from dataloader.dataset_create_gp import DATAOUT_DIR, work_dir



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


    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(work_dir,
                                     scenario_name)
        dataloader = dataloader_factory(scenario_path, direction=Direction.BOTH)


        THREAD_AWARE = True
        WINDOW_LENGTH = 1000
        NGRAM_LENGTH = 5

        ### building blocks
        syscall_name = SyscallName()
        int_embedding = IntEmbedding(syscall_name, scenario_name=scenario_name)
        ngram = Ngram([int_embedding], True, 5)
        nn = NearestNeighbour(ngram)

        ids = IDS(data_loader=dataloader,
                  resulting_building_block=nn,
                  create_alarms=True,
                  plot_switch=False)

        print("at evaluation:")

        results = ids.detect().get_results()
        pprint(results)

