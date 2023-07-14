

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
import pandas as pd

class Sequence_period(BuildingBlock):
    def __init__(self):

        super().__init__()
        self.start_time = 0
        self.seq_df = pd.DataFrame(columns=['ProcessName', 'TID', 'syscall'])

    def depends_on(self):
        return []

    def _calculate(self, syscall):

        return syscall[['ProcessName', 'TID', 'syscall']]
        # self.seq_df.append([syscall.process_name(), syscall.thread_id(), syscall.name()])
        # if self.start_time == 0:
        #     self.start_time = syscall.timestamp_unix_in_ns()
        #     return None
        #
        # current_time = syscall.timestamp_unix_in_ns()
        # if current_time - self.start_time > 1000000000:
        #     self.start_time = 0
        #     df_ret = self.seq_df.copy(deep=True)
        #     self.seq_df.drop(self.seq_df.index, inplace=True)
        #     return df_ret
        #
        # return None

    def new_recording(self):
        self.start_time = 0
        # self.seq_df.drop(self.seq_df.index, inplace=True)