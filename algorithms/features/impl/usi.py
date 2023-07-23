

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
import networkx as nx
from collections import Counter

class Usi(BuildingBlock):
    """
     calculate Un seen interface
    """

    def __init__(self, intput_block: BuildingBlock, ssc:BuildingBlock):
        """
        """
        super().__init__()
        # depands on Seen syscall list and sequence per period
        self._intput_block = intput_block
        self._dependency_list = [intput_block, ssc]
        self._ssc = ssc
        self.ssg_edges = []
    def depends_on(self):
        return self._dependency_list

    def _get_graph(self, g_edges):
        graph = nx.DiGraph((x, y, {'weight': v}) for (x, y), v in Counter(g_edges).items())
        return graph

    def fit(self):
        self._seen_syscalls = self._ssc.get_seen_sc()

    def _calculate(self, syscall):
        seq_df = self._intput_block.get_result(syscall)
        if seq_df is not None:
            seq_sc = list(seq_df['syscallInt'].unique())
            distinct_unseen_syscalls = (set(seq_sc) - self._seen_syscalls)
            if distinct_unseen_syscalls:
                seq_df.loc[~seq_df.syscall.isin(self._seen_syscalls), 'syscall'] = 'USN'

                syscalls_list = seq_df['syscallInt'].to_list()
                for i in range(len(syscalls_list) - 1):
                    self.ssg_edges.append((syscalls_list[i], syscalls_list[i + 1]))

                sequence_graph = self._get_graph(self.ssg_edges)

                usn_in_centrality = nx.in_degree_centrality(sequence_graph).get('USN')
                usn_out_centrality = nx.out_degree_centrality(sequence_graph).get('USN')

                if isinstance(usn_in_centrality, type(None)):
                    usn_in_centrality = 0

                if isinstance(usn_out_centrality, type(None)):
                    usn_out_centrality = 0

                return len(distinct_unseen_syscalls) * (usn_in_centrality + usn_out_centrality)

        return 0