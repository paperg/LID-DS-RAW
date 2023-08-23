

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
import pandas as pd

class Sequence_period(BuildingBlock):
    def __init__(self):

        super().__init__()
        # self.seq_df = pd.DataFrame(columns=['time', 'ProcessName', 'TID', 'syscallInt'])

    def depends_on(self):
        return []

    def _calculate(self, syscall):
        return syscall

    def new_recording(self):
        self.start_time = 0
        # self.seq_df.drop(self.seq_df.index, inplace=True)