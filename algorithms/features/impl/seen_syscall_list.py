

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall

class SeenSysC(BuildingBlock):
    """
     calculate Un seen interface
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.seen_syscalls = set()
        self._dependency_list = []
    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall):
        self.seen_syscalls = self.seen_syscalls | set(syscall['syscall'].unique())

    def _calculate(self, syscall):
        return None

    def get_seen_sc(self):
        if len(self.seen_syscalls) > 0:
            return self.seen_syscalls

        return None
