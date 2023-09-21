from torch_geometric.loader import GraphSAINTRandomWalkSampler
import torch

class VariantRandomWalkSamler(GraphSAINTRandomWalkSampler):
    def __init__(self, data, batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir=None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage,
                         save_dir, log, **kwargs)
        self.variant_idx=data.x[2]
    def _sample_nodes(self, batch_size):
        start=torch.tensor(self.variant_idx,dtype=torch.long).reshape((batch_size,))
        # start = torch.randint(0, self.N, (batch_size, ), dtype=torch.long)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)