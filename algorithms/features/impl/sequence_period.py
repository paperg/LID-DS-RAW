

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
        # return syscall[['ProcessName', 'TID', 'syscall']]
        # current_time = syscall.timestamp_unix_in_ns()
        # df_new_item = pd.DataFrame([[current_time, syscall.process_name(), syscall.thread_id(), syscall.name()]],
        #                    columns=['time', 'ProcessName', 'TID', 'syscall'])
        # self.seq_df = self.seq_df.append(df_new_item, ignore_index=True)
        #
        # if current_time - self.seq_df.at[0, 'time'] > 1000000000:
        #     result = self.seq_df.copy(deep=True)
        #     self.seq_df.drop(self.seq_df.index[[i for i in range(len(self.seq_df[self.seq_df['time'] < self.seq_df.at[0, 'time'] + 200000000]))]], inplace=True)
        #     self.seq_df.reset_index(drop=True, inplace=True)
        #
        #     return result
        #
        # return None

    def new_recording(self):
        self.start_time = 0
        # self.seq_df.drop(self.seq_df.index, inplace=True)