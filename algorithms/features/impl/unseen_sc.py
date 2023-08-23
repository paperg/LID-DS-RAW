from collections import deque

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class UnseenSystemCall(BuildingBlock):

    def __init__(self, input: BuildingBlock):
        super().__init__()
        # parameter
        self._input = input
        # internal data
        self._graphs = {}
        self._last_added_nodes = {}
        self._result_dict = {}

        # dependency list
        self._dependency_list = []
        self._dependency_list.append(self._input)

    def depends_on(self):
        return self._dependency_list

    def train_on(self, seq_df):

        if seq_df is not None:
            new_node = new_node[:2]
            # check for threads
            # do not use thread id for now
            tid = syscall.thread_id()
            # graph id
            pname = syscall.process_name()

            # check for graph
            # create a new graph for every process
            if pname not in self._graphs:
                self._graphs[pname] = nx.DiGraph()

            # check for last added node
            if tid not in self._last_added_nodes:
                self._last_added_nodes[tid] = None

            # finally add the input
            if self._last_added_nodes[tid] is None:
                self._graphs[pname].add_node(new_node)
            else:
                count = 0
                # edge already in graph? then update its freq.
                if self._graphs[pname].has_edge(self._last_added_nodes[tid], new_node):
                    count = self._graphs[pname].edges[self._last_added_nodes[tid], new_node]["f"]
                    # print(count)
                count += 1
                self._graphs[pname].add_edge(self._last_added_nodes[tid], new_node, f=count)
            self._last_added_nodes[tid] = new_node

    def fit(self):
        print(f"got {len(self._graphs)} graphs")
        s_n = 0
        s_e = 0
        for g in self._graphs.values():
            s_n += g.number_of_nodes()
            s_e += g.number_of_edges()
        print(f"with in sum: {s_n} nodes and {s_e} edges")
        for g in self._graphs.values():
            for source_node in g.nodes:
                sum_out = 0
                for s, t, data in g.out_edges(nbunch=source_node, data=True):
                    f = data["f"]
                    sum_out += f
                for s, t, data in g.out_edges(nbunch=source_node, data=True):
                    f = data["f"]
                    g.add_edge(s, t, f=f, p=f / sum_out)

    def _calculate(self, syscall: Syscall):
        """
        calculates transition probability
        """
        # the new node
        new_node = self._input.get_result(syscall)
        if new_node is not None:
            # the thread id
            tid = syscall.thread_id()
            pname = syscall.process_name()

            if tid in self._last_added_nodes:
                # is the result already calculated?
                s = self._last_added_nodes[tid]
                t = new_node
                edge = tuple([s, t])
                if pname not in ['mysqld', 'apache2']:
                    print(f'{pname} is invalid')
                if edge in self._result_dict[pname]:
                    self._last_added_nodes[tid] = new_node
                    return self._result_dict[pname][edge]
                else:
                    # was not the first node for this tid
                    transition_probability = 0

                    g = self._graphs[pname]
                    if g.has_edge(s, t):
                        transition_probability += g[s][t]["p"]
                    self._result_dict[pname][edge] = transition_probability
                    self._last_added_nodes[tid] = new_node
                    return transition_probability
            else:
                self._last_added_nodes[tid] = new_node
                return None
        else:
            return None

    def new_recording(self):
        self._last_added_nodes = {}