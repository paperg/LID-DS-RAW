

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
import networkx as nx
from collections import Counter

class Scg_Seq(BuildingBlock):
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
    def depends_on(self):
        return self._dependency_list

    def _get_graph(self, g_edges):
        graph = nx.DiGraph((x, y, {'f': v}) for (x, y), v in Counter(g_edges).items())
        return graph

    def _calculate(self, syscall):
        seq_df = self._intput_block.get_result(syscall)
        mysqld_sequence_graph = None
        apache_sequence_graph = None
        ssg_edges = []

        if seq_df is not None:
            syscalls_list = seq_df[seq_df['ProcessName'] == 'mysqld']['syscall'].to_list()
            for i in range(len(syscalls_list) - 1):
                ssg_edges.append((syscalls_list[i], syscalls_list[i + 1]))
            if len(ssg_edges) > 0:
                mysqld_sequence_graph = self._get_graph(ssg_edges)

            syscalls_list = seq_df[seq_df['ProcessName'] == 'apache2']['syscall'].to_list()
            for i in range(len(syscalls_list) - 1):
                ssg_edges.append((syscalls_list[i], syscalls_list[i + 1]))
            if len(ssg_edges) > 0:
                apache_sequence_graph = self._get_graph(ssg_edges)

        return tuple(mysqld_sequence_graph, apache_sequence_graph)