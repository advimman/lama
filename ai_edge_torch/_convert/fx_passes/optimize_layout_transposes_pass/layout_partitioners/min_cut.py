# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Min cut solver for partitioning the graph module into NHWC and non-NHWC subgraphs."""

import collections
import dataclasses

from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_check  # NOQA
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import layout_mark  # NOQA
import numpy as np
import scipy
import torch


def can_partition(graph_module: torch.fx.GraphModule):
  """Returns true if the input graph_module can be partitioned by min cut solver

  in a reasonable time.

  The min cut solver implements O(|V|^2|E|) Dinic's algorithm, which may
  take a long time to complete for large graph module. This function determines
  whether the graph module can be partitioned by the graph module size.
  """
  graph = graph_module.graph
  n_nodes = len(graph.nodes)
  n_edges = sum(len(n.users) for n in graph.nodes)

  # According to the experiments our model set, |V| < 2000 can
  # be partitioned generally in a reasonable time.
  return n_nodes**2 * n_edges < 2000**3


class MinCutSolver:
  # A number that is large enough but can fit into int32 with all computations
  # in the maximum flow.
  INF_COST = 1 << 28

  def __init__(self):
    self._edges_map = collections.defaultdict(dict)
    self._obj_to_node = {}
    self._node_to_obj = {}
    self._nodes_cnt = 0

    self.source = self._next_nid()
    self.sink = self._next_nid()

  def _next_nid(self):
    nid = self._nodes_cnt
    self._nodes_cnt += 1
    return nid

  @property
  def nodes(self):
    return list(range(self._nodes_cnt))

  @property
  def edges_map(self):
    return self._edges_map

  @property
  def edges(self):
    return [
        [n, m, cost]
        for n, next_nodes in self._edges_map.items()
        for m, cost in next_nodes.items()
    ]

  @property
  def graph(self):
    edges = np.array(self.edges)
    return scipy.sparse.csr_matrix(
        (
            np.minimum(edges[:, 2], MinCutSolver.INF_COST),
            (edges[:, 0], edges[:, 1]),
        ),
        shape=(self._nodes_cnt, self._nodes_cnt),
        dtype=np.int32,
    )

  def get_nid(self, obj=None):
    if obj is None:
      return self._next_nid()

    nid = self._obj_to_node.get(obj)
    if nid is None:
      nid = self._next_nid()

      self._obj_to_node[obj] = nid
      self._node_to_obj[nid] = obj
    return nid

  def get_obj(self, nid: int):
    return self._node_to_obj.get(nid, None)

  def add_edge(self, a_id: int, b_id: int, cost: int):
    assert isinstance(cost, int)
    self._edges_map[a_id][b_id] = cost

  def solve(self):
    flow = scipy.sparse.csgraph.maximum_flow(
        self.graph, self.source, self.sink, method="dinic"
    ).flow

    # Max-flow min-cut theorem: find min-cuts in the residual network.
    ds = scipy.cluster.hierarchy.DisjointSet(self.nodes)
    for n, m, cost in self.edges:
      if abs(flow[n, m]) < cost:
        ds.merge(n, m)

    residual_reachable_nodes = ds.subset(self.source)

    cuts = set()
    for n, m, cost in self.edges:
      if n in residual_reachable_nodes and m not in residual_reachable_nodes:
        cuts.add((n, m))

    return cuts


@dataclasses.dataclass(frozen=True)
class MultiUsersDummyNode:
  src: torch.fx.Node


def partition(graph_module: torch.fx.GraphModule):
  """Partition the graph module into NHWC and non-NHWC subgraphs, and mark

  nodes in the NHWC partitions.

  Implements O(|V|^2|E|) min-cut (optimal) partitioning algorithm.
  """
  graph = graph_module.graph

  mc_solver = MinCutSolver()
  for fx_node in graph.nodes:
    if layout_mark.is_const_node(fx_node):
      continue

    nid = mc_solver.get_nid(fx_node)
    if fx_node.op in ("placeholder", "output"):
      # All inputs and outputs are not NHWCable nodes in the graph,
      # connected to source S directly with inf cost to cut
      mc_solver.add_edge(mc_solver.source, nid, cost=MinCutSolver.INF_COST)
    elif not layout_check.can_be_nhwc(fx_node):
      # All not NHWCable nodes are connected to source S directly,
      # with inf cost to cut.
      mc_solver.add_edge(mc_solver.source, nid, cost=MinCutSolver.INF_COST)
    elif layout_check.must_be_nhwc(fx_node):
      # All must be NHWC nodes are connected to sink T directly,
      # with inf cost to cut
      mc_solver.add_edge(nid, mc_solver.sink, cost=MinCutSolver.INF_COST)

    cut_cost = 10  # set 10 to be a unit of cut cost
    if fx_node.target in (torch.ops.aten.mean.default, torch.ops.aten.mean.dim):
      # TFLite converter cannot fuse the lowering of (tpos-mean) but (mean-tpos)
      # when it applies on the feature dimensions. Therefore decreasing the cut
      # cost for aten.mean's out-going edges to favor having a cut (transpose)
      # after the node than before when the number of transposes are equal.
      # TODO: Remove this rule when converter has fuse rule for tpos-mean.
      cut_cost = 9

    if len(fx_node.users) > 1:
      # If a node's (A1) output is used by multiple nodes (B1, B2, B3, ...),
      # the cost to split A1 and Bs into different partitions would just be 1
      # transpose. So we need to introduce a dummy node between A1 and Bs in the
      # min-cut graph to reflect the fact that disconnecting them doesn't
      # introduce multiple transposes.
      dummy_nid = mc_solver.get_nid(MultiUsersDummyNode(fx_node))
      mc_solver.add_edge(nid, dummy_nid, cost=cut_cost)
      mc_solver.add_edge(dummy_nid, nid, cost=cut_cost)
      nid = dummy_nid

    for user in fx_node.users:
      # All the other nodes and edges in the model graph are scattered
      # and connected as is in the new graph, with 1 cost to cut an edge.
      user_id = mc_solver.get_nid(user)
      mc_solver.add_edge(nid, user_id, cost=cut_cost)
      mc_solver.add_edge(user_id, nid, cost=cut_cost)

  cuts = mc_solver.solve()

  # Find nodes that is connected to sink after the min-cut and mark as NHWC.
  ds = scipy.cluster.hierarchy.DisjointSet(mc_solver.nodes)
  for n, m, cost in mc_solver.edges:
    if (n, m) in cuts or (m, n) in cuts:
      continue
    ds.merge(n, m)
  assert not ds.connected(mc_solver.source, mc_solver.sink)

  for nid in mc_solver.nodes:
    if ds.connected(nid, mc_solver.source):
      continue

    obj = mc_solver.get_obj(nid)
    if obj is None:
      continue
    if isinstance(obj, MultiUsersDummyNode):
      continue

    assert isinstance(obj, torch.fx.Node)
    layout_mark.mark_as_nhwc_node(obj)

  graph_module.recompile()
  return graph_module
