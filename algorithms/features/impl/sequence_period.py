
from algorithms.building_block import BuildingBlock
import pandas as pd

class Sequence_period(BuildingBlock):
    def __init__(self):
        super().__init__()
        self._period_df = []
    def depends_on(self):
        return []

    def train_on(self, sequences):
        if sequences is not None:
            one_file = []
            for seq in sequences:
                one_file.append(seq[['time', 'syscall','Params']])
            self._period_df.append(one_file)

    def _calculate(self, syscall):
        return self._period_df

    def _get_result(self):
        return self._period_df

    def new_recording(self):
        self._period_df = []