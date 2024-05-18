"""
example script for running GP
"""
import os
import sys
sys.path.append('.')
import time

from pprint import pprint
from datetime import datetime
from dataloader.direction import Direction
from dataloader.dataloader_factory import dataloader_factory

from algorithms.persistance import save_to_mongo
from algorithms.features.impl.array_max_threshold import ArrayMaxThreshold
from algorithms.ids import IDS
from algorithms.decision_engines.tranAD import Transformer_ad
from algorithms.decision_engines.CHBS_Model import CHBS_ad
from dataloader.dataset_create_gp import DATAOUT_DIR, work_dir
from flask import Flask, render_template_string

app = Flask(__name__)

# 定义 HTML 模板
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataFrames</title>
    <style>
        table {
            border-collapse: collapse;
            width: 50%;
            margin: auto; 
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        th, td {
            border: 1px solid black;
            padding: 10px;
            text-align: left;
        }
        th {
            font-weight: bold;
        }
        tr.normal {
            background-color: white;
        }
        tr.red {
            background-color: #FFCCCC;
        }
        tr.blue {
            background-color: #CCE5FF;
        }
    </style>
</head>
<body>
    {% for name, dataframe in dataframes.items() %}
        <h2>{{ name }}</h2>
        <table>
            <thead>
                <tr>
                    {% for col in dataframe.columns %}
                        <th>{{ col }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for index, row in dataframe.iterrows() %}
                    <tr class="{{ 'red' if row['type'] == 3 else 'blue' if row['type'] == 1 else 'normal' }}">
                        {% for col in dataframe.columns %}
                            <td>{{ '异常' if row[col] == 1 else '入侵时刻' if row[col] == 3 else '正常' if row[col] == 0 else row[col] }}</td>
                        {% endfor %}
                    </tr>
                {% endfor %}
            </tbody>
        </table>
    {% endfor %}
</body>
</html>
"""

web_data = {}
@app.route('/')
def index():
    return render_template_string(template, dataframes=web_data)

if __name__ == '__main__':

    LID_DS_VERSION_NUMBER = 1
    LID_DS_VERSION = [
            "LID-DS-2019",
            "LID-DS-2021"
            ]

    # scenarios ordered by training data size asc
    SCENARIOS = [
        "Bruteforce_CWE-307",
        "Juice-Shop",
        "CVE-2020-13942",
        "CWE-89-SQL-injection",
        "CVE-2012-2122",
        "CVE-2018-3760",
        "CVE-2020-23839",
        "CVE-2020-9484",
        "EPS_CWE-434",
        "CVE-2014-0160",
        "CVE-2017-7529",
        "Bruteforce_CWE-307",
        "CVE-2019-5418",
        "PHP_CWE-434",
        "project",
    ]
    SCENARIO_RANGE = SCENARIOS[0:1]

    # config
    # Transformer
    EPOCH = 120
    # DROPOUT=0.4
    DROPOUT = 0.2
    BATCH_SIZE = 128

    HIDDEN_LAYERS = 3
    NUM_HEAD = 8
    # True False
    USE_TIME_DELTA = False
    USE_PTID = False
    USE_USA = True
    USE_USS = True
    USE_FREQ = True
    USE_RET = True
    USE_SC_MAX_PARAMS = True
    # model
    USE_AE2 = True
    USE_TRANSFORMER=False

    use_dict = {'use_timedelta': USE_TIME_DELTA, 'use_ptidfreq': USE_PTID, 'use_usa': USE_USA, 'use_usc': USE_USS,
                'use_freq': USE_FREQ, 'use_sc_max_params': USE_SC_MAX_PARAMS, 'use_ret':USE_RET,'mode_use_ae2': USE_AE2, 'use_transformer':USE_TRANSFORMER}
    # getting the LID-DS base path from argument or environment variable
    LID_DS_BASE_PATH = 'K:/hids'

    for scenario_name in SCENARIO_RANGE:
        scenario_path = os.path.join(work_dir,
                                     scenario_name)
        # data loader for scenario
        dataloader = dataloader_factory(scenario_path, direction=Direction.CLOSE)

        element_size = 39 * USE_TIME_DELTA + USE_PTID * 4 + USE_USA + USE_RET + USE_USS + USE_FREQ + USE_SC_MAX_PARAMS * 12
        input_dim = element_size

        USE_MINE = True
        if USE_MINE:
            model_path = f'{LID_DS_BASE_PATH}/Models/{scenario_name}/model/' \
                         f'epoch{EPOCH}_' \
                         f'batch{BATCH_SIZE}_' \
                         f'drop{DROPOUT}_' \
                         f'hidLay{HIDDEN_LAYERS}_' \
                         f'NumH{NUM_HEAD}_' \
                         f'TimeDel_{USE_TIME_DELTA}_' \
                         f'PTID_{USE_PTID}_' \
                         f'Usa_{USE_USA}_' \
                         f'Uss{USE_USS}_' \
                         f'UFreq_{USE_FREQ}_' \
                         f'URET_{USE_RET}' \
                         f'AE2_{USE_AE2}_' \
                         f'SC_MAX_{USE_SC_MAX_PARAMS}.model'

            AD_Module = Transformer_ad(
                        input_dim=input_dim,
                        epochs=EPOCH,
                        dropout=DROPOUT,
                        num_head=NUM_HEAD,
                        hidden_layers=HIDDEN_LAYERS,
                        batch_size=BATCH_SIZE,
                        model_path=model_path,
                        use_dict=use_dict,
                        scenario_path=f'{LID_DS_BASE_PATH}/Data/{scenario_name}')
        else:
            model_path = f'{LID_DS_BASE_PATH}/Models/{scenario_name}/CHBS_model/' \
                         f'epoch{EPOCH}_' \
                         f'batch{BATCH_SIZE}.model'
            AD_Module = CHBS_ad(input_dim=input_dim,
                              epochs=EPOCH,
                              batch_size=BATCH_SIZE,
                              use_dict=use_dict,
                              model_path=model_path,
                              scenario_path=f'{LID_DS_BASE_PATH}/Data/{scenario_name}')

        decider = ArrayMaxThreshold(AD_Module,
                                    use_timedelta=USE_TIME_DELTA,
                                    use_ret=USE_RET,
                                    use_usa=USE_USA,
                                    use_usc=USE_USS,
                                    use_freq=USE_FREQ,
                                    use_sc_max_params=USE_SC_MAX_PARAMS,
                                    use_my_mode = USE_MINE,
                                    use_transformer=USE_TRANSFORMER)
        # define the used features
        ids = IDS(data_loader=dataloader,
                  resulting_building_block=decider,
                  create_alarms=False,
                  plot_switch=False)

        start = time.time()
        # # detection
        performance = ids.detect_batchs(AD_Module)
        results = performance.get_results()
        end = time.time()
        detection_time = (end - start) / 60  # in min
        results['scenario'] = scenario_name
        # results['Model'] = 'Mine_test_no_trasnformer'
        results['Model'] = 'Mine_Final'
        results['date'] = str(datetime.now().date())
        results['detection_time'] = detection_time
        pprint(results)
        # web_data = ids.web_dic
        # app.run(debug=False)
        # save_to_mongo(results)
