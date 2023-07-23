from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class ProcessNameAndRet(BuildingBlock):

    def __init__(self, intembed: list):
        super().__init__()
        self._fileter_sc = ['write', 'mmap', 'brk', 'writev', 'read', 'sendto', 'recvfrom', 'lseek', 'pwrite', 'pread', 'clone']
        self._intembed = intembed
    def _calculate(self, syscall: Syscall):
        """
        calculate name of process
        """
        scint = self._intembed._calculate(syscall)
        retvalue = 0
        return_value_string = syscall.param('res')
        if return_value_string is not None:
            try:
                retvalue = int(return_value_string.split('(')[0])
                if syscall.name()  in self._fileter_sc:
                    if retvalue > 0:
                        retvalue = 1
            except ValueError:
                try:
                    retvalue = int(return_value_string.split('(')[0], 16)
                    if syscall.name() in self._fileter_sc:
                        if retvalue > 0:
                            retvalue = 1
                except ValueError:
                    print('valuse error %s' % return_value_string)
                    pass

        return tuple([scint, retvalue])

    def depends_on(self):
        return []
