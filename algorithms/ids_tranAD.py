"""
example script for running LSTM
"""
import os
import time

from pprint import pprint
from datetime import datetime
from dataloader.direction import Direction
from dataloader.dataloader_factory import dataloader_factory

from algorithms.persistance import save_to_mongo
from algorithms.features.impl.array_max_threshold import ArrayMaxThreshold
from algorithms.ids import IDS
from algorithms.decision_engines.tranAD import Transformer_ad
from dataloader.dataset_create_gp import DATAOUT_DIR, work_dir

if __name__ == '__main__':

    LID_DS_VERSION_NUMBER = 1
    LID_DS_VERSION = [
            "LID-DS-2019",
            "LID-DS-2021"
            ]

    # scenarios ordered by training data size asc
    SCENARIOS = [
        "Bruteforce_CWE-307",
        "CVE-2017-7529",
        "CWE-89-SQL-injection",


        "CVE-2012-2122",

      "CVE-2018-3760",
      "CVE-2019-5418",
      "PHP_CWE-434",
      "EPS_CWE-434",

      "CVE-2014-0160",
      "CVE-2020-23839",
      "ZipSlip",
      "CVE-2020-9484",
      "Juice-Shop",
      "CVE-2020-13942",
      "CVE-2017-12635_6"
    ]
    SCENARIO_RANGE = SCENARIOS[0:1]

    # config
    # Transformer
    EPOCH=30
    DROPOUT=0.5
    BATCH_SIZE = 32

    HIDDEN_LAYERS = 1
    NUM_HEAD = 2
    # True False
    USE_TIME_DELTA = False
    USE_PTID = False
    USE_USA = True
    USE_USS = True
    USE_FREQ = False

    # getting the LID-DS base path from argument or environment variable
    LID_DS_BASE_PATH = 'L:/hids'

    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(work_dir,
                                     scenario_name)
        # data loader for scenario
        dataloader = dataloader_factory(scenario_path, direction=Direction.CLOSE)

        element_size = 39 * USE_TIME_DELTA + USE_PTID * 4 + USE_USA + USE_USS + USE_FREQ

        input_dim = element_size

        model_path = f'{LID_DS_BASE_PATH}/Models/{scenario_name}/model/'\
            f'epoch{EPOCH}' \
            f'batch{BATCH_SIZE}' \
            f'drop{DROPOUT}' \
            f'hidLay{HIDDEN_LAYERS}' \
            f'NumH{NUM_HEAD}' \
            f'TimeDel_{USE_TIME_DELTA}' \
            f'PTID_{USE_PTID}' \
            f'Usa_{USE_USA}' \
            f'Uss{USE_USS}' \
            f'UFreq_{USE_FREQ}.model'

        transAD = Transformer_ad(
                    input_dim=input_dim,
                    epochs=EPOCH,
                    dropout=DROPOUT,
                    num_head=NUM_HEAD,
                    hidden_layers=HIDDEN_LAYERS,
                    batch_size=BATCH_SIZE,
                    model_path=model_path,
                    use_timedelta=USE_TIME_DELTA,
                    use_ptidfreq=USE_PTID,
                    use_usa=USE_USA,
                    use_usc=USE_USS,
                    use_freq=USE_FREQ,
                    scenario_path=f'{LID_DS_BASE_PATH}/Data/{scenario_name}')

        decider = ArrayMaxThreshold(transAD,
                                    use_timedelta=USE_TIME_DELTA,
                                    use_ptidfreq=USE_PTID,
                                    use_usa=USE_USA,
                                    use_usc=USE_USS,
                                    use_freq=USE_FREQ)
        # define the used features
        ids = IDS(data_loader=dataloader,
                  resulting_building_block=decider,
                  create_alarms=False,
                  plot_switch=False)

        start = time.time()
        # # detection
        performance = ids.detect_batchs(transAD)
        results = performance.get_results()
        end = time.time()
        detection_time = (end - start) / 60  # in min
        pprint(results)

        results = transAD.get_results(results)
        results['scenario'] = scenario_name
        results['date'] = str(datetime.now().date())
        results['detection_time'] = detection_time
        results['Model'] = model_path

        # save_to_mongo(results)
