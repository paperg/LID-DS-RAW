

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall

class TidNumber(BuildingBlock):
    """
     calculate Un seen interface
    """

    def __init__(self, intput_block: BuildingBlock):
        """
        """
        super().__init__()
        # depands on Seen syscall list and sequence per period
        self._intput_block = intput_block
        self._dependency_list = [intput_block]
        self.ssg_edges = []
    def depends_on(self):
        return self._dependency_list

    def _calculate(self, syscall):
        seq_df = self._intput_block.get_result(syscall)
        if seq_df is not None:
            maxval = seq_df['TID'].value_counts().max()
            if maxval > 10000:
                print(f'Max Tid value {maxval}')

            return maxval // 10000

        return 0