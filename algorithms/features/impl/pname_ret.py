from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
import numpy as np

class ProcessNameAndRet(BuildingBlock):

    def __init__(self, intembed: list):
        super().__init__()
        self._fileter_sc = ['write', 'mmap', 'brk', 'writev', 'read', 'sendto', 'recvfrom', 'lseek', 'pwrite', 'pread' 'sendmsg', 'recvmsg', 'sendfile']
        self._intembed = intembed
    def _calculate(self, syscall: Syscall):
        """
        calculate name of process
        """
        scint = self._intembed._calculate(syscall)
        retvalue = 0
        real_retvalue = 0
        return_value_string = syscall.param('res')
        if return_value_string is not None:
            try:
                real_retvalue = int(return_value_string.split('(')[0])
                # if syscall.name() in self._fileter_sc:
                #     if real_retvalue > 0:
                #         retvalue = 1
                #     elif real_retvalue < 0:
                #         real_retvalue = -1
            except ValueError:
                try:
                    real_retvalue = int(return_value_string.split('(')[0], 16)
                    # if syscall.name() in self._fileter_sc:
                    #     if real_retvalue > 0:
                    #         retvalue = 1
                    #     elif real_retvalue < 0:
                    #         real_retvalue = -1
                except ValueError:
                    print('valuse error %s' % return_value_string)
                    pass

            if retvalue == 0:
                if real_retvalue >= 0:
                    retvalue = 0
                else:
                    retvalue = -1

        return tuple([scint, retvalue, real_retvalue])

    def _calculate_m(self, scname: str, row_rets:str):
        """
        calculate name of process
        """
        scint = self._intembed._calculate(scname)
        retvalue = 0
        real_retvalue = 0

        if row_rets is np.nan:
            return tuple([scint, retvalue, real_retvalue])

        name, return_value_string = row_rets.split('=')

        if return_value_string is not None:
            try:
                real_retvalue = int(return_value_string.split('(')[0])
                # if scname in self._fileter_sc:
                #     if real_retvalue > 0:
                #         retvalue = 1
            except ValueError:
                try:
                    real_retvalue = int(return_value_string.split('(')[0], 16)
                    # if scname in self._fileter_sc:
                    #     if real_retvalue > 0:
                    #         retvalue = 1
                except ValueError:
                    print('valuse error %s' % return_value_string)
                    pass

            if retvalue == 0:
                if real_retvalue >= 0:
                    retvalue = 0
                else:
                    retvalue = -1

            # if name != 'res':
            #     retvalue = 0
            #     if name not in ['fd', 'uid', 'euid', 'egid', 'gid']:
            #         print(f'{name} is not res !!')

        return tuple([scint, retvalue, real_retvalue])
    def depends_on(self):
        return []
