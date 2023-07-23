

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall
import networkx as nx
import pandas as pd

class Syscall_Frequency(BuildingBlock):
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
        self._apache2_scg = nx.read_multiline_adjlist("K:/hids/LID-DS-master/tools/out/CWE-89-SQL-injection/SCG/apache2_scg",
                                                delimiter='|', create_using=nx.DiGraph)
        self._mysqld_scg = nx.read_multiline_adjlist("K:/hids/LID-DS-master/tools/out/CWE-89-SQL-injection/SCG/mysqld_scg",
                                                delimiter='|', create_using=nx.DiGraph)
    def depends_on(self):
        return self._dependency_list

    def _calculate(self, syscall):
        result = []
        mysqld_sequence_graph, apache_sequence_graph = self._intput_block.get_result(syscall)
        if mysqld_sequence_graph is not None:
            edge_df = pd.DataFrame(columns=[0, 1])
            for (u, v, wt) in mysqld_sequence_graph.edges.data('f'):
                if wt > 2:
                    if self._mysqld_scg.has_edge(u, v):
                        edge_df.loc[len(edge_df.index)] = [self._mysqld_scg.edges[u, v]["f"], wt]
                    else:
                        print(f'mysqld_scg has not age {u} to {v} wt {wt}')


            largest = 5 if len(edge_df) > 5 else len(edge_df)

            if largest == 5:
                result.extend(list(edge_df[0].astype(int).nlargest(5).index == edge_df[1].astype(int).nlargest(5).index))

            for (u, v, wt) in apache_sequence_graph.edges.data('f'):
                if wt > 2:
                    if self._apache2_scg.has_edge(u, v):
                        edge_df.loc[len(edge_df.index)] = [self._apache2_scg.edges[u, v]["f"], wt]
                    else:
                        print(f'apache2_scg has not age {u} to {v} wt {wt}')

            largest = 5 if len(edge_df) > 5 else len(edge_df)

            if largest == 5:
                result.extend(list(edge_df[0].astype(int).nlargest(5).index == edge_df[1].astype(int).nlargest(5).index))

        return result