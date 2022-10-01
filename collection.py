import torch
from torch import Tensor
from typing import Tuple


def set_check(tensor1: Tensor, tensor2: Tensor):
    if len(tensor1.shape) > 1 or len(tensor2.shape) > 1:
        raise NotImplementedError(
            f"set operations only support 1d tensor, but got {len(tensor1.shape)} for tensor1 "
            f"and {len(tensor2.shape)} for tensor2"
        )


class Collections:
    @staticmethod
    def set_intersection(tensor1: Tensor, tensor2: Tensor) -> Tensor:
        mask = tensor1 == tensor2.view(-1, 1)
        keep = mask.any(dim=0)
        return tensor1[keep]

    @staticmethod
    def set_difference(tensor1: Tensor, tensor2: Tensor) -> Tensor:
        set_check(tensor1, tensor2)
        mask = tensor1 == tensor2.view(-1, 1)
        keep = ~mask.any(dim=0)
        return tensor1[keep]

    @staticmethod
    def set_union(tensor1: Tensor, tensor2: Tensor) -> Tensor:
        set_check(tensor1, tensor2)
        return torch.cat([tensor1, tensor2]).unique()

    @staticmethod
    @torch.jit.script
    def re_mapping(nodes: Tensor, edges: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            nodes: shape [n], example: [-10323333, -1022299, 1092222]
            edges: shape [2, m] or any, example: [[-10323333, -1022299], [-10323333, 1092222], [-1022299, 1092222]]
        Return:
            mapped_nodes: shape [n], example: [0, 1, 2]
            mapped_edges: shape [2, m] or any, example: [[0, 1], [0, 2], [1, 2]]

        Vanilla Implementation:
            mapper = {x.item(): i for i, x in enumerate(nodes)}
            mapped_nodes = torch.tensor(list(mapper.keys()))
            mapped_edges = torch.tensor([mapper[x.item()] for x in edges.flatten()]).reshape(2, -1)
            return mapped_nodes, mapped_edges
        """
        nodes = nodes.sort()[0]
        mapped_nodes = torch.arange(nodes.numel())
        flatten_edges = edges.flatten()
        sorted_edges, sorted_edge_indices = torch.sort(flatten_edges)

        # count the sorted nodes, and it will be used in repeat_interleave
        mask_edges = sorted_edges.roll(1, dims=0)
        mask_edges = mask_edges != sorted_edges
        mask_edges[0] = True
        num_nodes = mask_edges.nonzero()[:, 0]
        unique_edge_nodes = sorted_edges[num_nodes]
        num_nodes[0] = sorted_edges.numel()
        num_nodes = num_nodes.roll(-1, dims=0)
        num_nodes[1:] = num_nodes[1:] - num_nodes[:-1]

        indices = torch.searchsorted(nodes, unique_edge_nodes)
        unique_edge_nodes = mapped_nodes[indices]

        sorted_edges = unique_edge_nodes.repeat_interleave(num_nodes, dim=0)
        mapped_edges = torch.zeros_like(flatten_edges)
        mapped_edges[sorted_edge_indices] = sorted_edges

        return mapped_nodes, mapped_edges.view(edges.shape)
