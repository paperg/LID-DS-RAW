from dataloader.syscall import Syscall

from algorithms.building_block import BuildingBlock
from algorithms.features.impl.syscall_name import SyscallName
import torch
import os
import json

DEFAULT_DIR= 'K:/hids/Models'
# 系统调用 name 转为 int
class IntEmbedding(BuildingBlock):
    """
        convert system call name to unique integer
        
        Params:
        building_block: BB which should be embedded as int
    """

    def __init__(self, building_block: BuildingBlock = None, scenario_name:str = 'default'):
        super().__init__()
        self._syscall_dict = {}
        self.need_train = True
        self._save_dir = os.path.join(DEFAULT_DIR,  scenario_name)
        self.save_path = os.path.join(self._save_dir, 'IntEmbedding_model')
        if building_block is None:
            building_block = SyscallName()
        self._dependency_list = [building_block]

        if os.path.exists(self._save_dir) is not True:
            os.mkdir(self._save_dir)

        if os.path.exists(self.save_path):
            self._syscall_dict = torch.load(self.save_path)
            self.need_train = False
            print('Load IntEmbedding Model From %s' % self.save_path)

    def depends_on(self):
        return self._dependency_list

    def is_needtrain(self):
        return self.need_train

    # 将所有系统调用存入 _syscall_dict[] key : name, value : index
    def train_on(self, syscall: Syscall):
        """
            takes one syscall and assigns integer
            integer is current length of syscall_dict
            keep 0 free for unknown syscalls
        """
        if not self.need_train:
            return

        bb_value = self._dependency_list[0].get_result(syscall)
        if bb_value not in self._syscall_dict:
            # index start from 1
            self._syscall_dict[bb_value] = len(self._syscall_dict) + 1

    def fit(self):
        if self.need_train:
            torch.save(self._syscall_dict, self.save_path)
            with open(os.path.join(self._save_dir, 'intEmbedding.json'), 'w') as f:
                json.dump(self._syscall_dict, f, indent=4, ensure_ascii=True, sort_keys=False)
            print("Save IntEmbedding Model To %s" % self.save_path)

    # 获取系统调用 name, 根据字典 _syscall_dict get index, or retuen 0
    def _calculate(self, syscall: Syscall):
        """
            transforms given building_block to integer
        """
        bb_value = self._dependency_list[0].get_result(syscall)
        try:
            sys_to_int = self._syscall_dict[bb_value]
        except KeyError:
            sys_to_int = len(self._syscall_dict) + 1
            print('KeeyError %s %d' % (bb_value, sys_to_int))
            self._syscall_dict[bb_value] = sys_to_int

        return sys_to_int

    def _calculate(self, bb_value: str):
        """
            transforms given building_block to integer
        """
        try:
            sys_to_int = self._syscall_dict[bb_value]
        except KeyError:
            sys_to_int = len(self._syscall_dict) + 1
            self._syscall_dict[bb_value] = sys_to_int
            # print('Key Error %s %d' % (bb_value, sys_to_int))
        return sys_to_int

class IntEmbeddingConcat(BuildingBlock):
    """
        convert input features to unique integers over all building blocks
        results are concatenated in the order of the building blocks
    """

    def __init__(self, building_blocks: list[BuildingBlock]):
        super().__init__()
        self._encoding_dict = {idx: {} for idx in range(len(building_blocks))}
        self._dependency_list = building_blocks
        self._size = None

    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall):
        """
            takes one feature and assigns it an integer
            the integer is current length of encoding_dict per building_block
            keep 0 free for unknown syscalls
        """
        if self._size is not None:
            return
        for idx, bb in enumerate(self._dependency_list):
            bb_value = bb.get_result(syscall)
            if isinstance(bb_value, (list, tuple)):
                for value in bb_value:
                    if value not in self._encoding_dict[idx]:
                        self._encoding_dict[idx][value] = len(self._encoding_dict[idx]) + 1
            else:
                if bb_value not in self._encoding_dict[idx]:
                    self._encoding_dict[idx][bb_value] = len(self._encoding_dict[idx]) + 1

    def fit(self):
        """ offset encodings and update size """
        if self._size is not None:
            return

        offset = 0
        for idx, encoding_dict in self._encoding_dict.items():
            for key, value in encoding_dict.items():
                encoding_dict[key] = value + offset
            offset += len(encoding_dict)
        self._size = offset + 1  # +1 for unknown

    def _calculate(self, syscall: Syscall):
        """
            transforms given building_block to integer
        """
        result = []
        for idx, bb in enumerate(self._dependency_list):
            bb_value = bb.get_result(syscall)
            if isinstance(bb_value, (list, tuple)):
                result.extend(self._encoding_dict[idx].get(value, 0) for value in bb_value)
            else:
                result.append(self._encoding_dict[idx].get(bb_value, 0))
        return tuple(result)

    def __len__(self):
        return self._size