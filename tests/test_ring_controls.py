"""Ring-size controls see center fate and counts, never neighborhood fate."""
import unittest
import torch
from torch_geometric.data import Data
from src.models.ring_mlp import RingSizeMLP
from src.data.ring_features import attach_precomputed_ring_features
from src.artifacts.checkpoints import build_model, model_spec


class RingControlTests(unittest.TestCase):
    def test_neighbor_fates_and_fraction_cache_cannot_affect_center(self):
        torch.manual_seed(12)
        x = torch.eye(3).repeat(3, 1)
        a = torch.arange(len(x) - 1)
        edges = torch.stack([torch.cat([a, a + 1]), torch.cat([a + 1, a])])
        graph = Data(x=x, edge_index=edges, global_feat=torch.tensor([[.2]]))
        graph.batch = torch.zeros(len(x), dtype=torch.long)
        for depth in (0, 1, 2, 4):
            with self.subTest(depth=depth):
                model = RingSizeMLP(3, k_hops=depth, hidden_dim=8, dropout=0., norm='layer',
                                    global_dim=1, use_center_markers=True).eval()
                rebuilt = build_model(model_spec(model))
                rebuilt.load_state_dict(model.state_dict(), strict=True)
                cached = graph.clone()
                attach_precomputed_ring_features([cached], k_hops=4, inplace=True)
                edited = cached.clone()
                edited.x[1:] = torch.randn_like(edited.x[1:]) * 100
                edited.x_ring.fill_(float('nan'))  # Any access to ring fractions must fail the equality.
                encoder_inputs = []
                hook = model.encoder.register_forward_pre_hook(lambda module, args: encoder_inputs.append(args[0].clone()))
                with torch.no_grad():
                    original, hidden = model(graph.x, graph.edge_index, data=graph)
                    precomputed, cached_hidden = model(cached.x, cached.edge_index, data=cached)
                    changed, edited_hidden = model(edited.x, edited.edge_index, data=edited)
                hook.remove()
                torch.testing.assert_close(original[0], precomputed[0], rtol=0, atol=0)
                torch.testing.assert_close(hidden, cached_hidden, rtol=0, atol=0)
                for before, after in zip(original, changed):
                    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
                torch.testing.assert_close(hidden[0], edited_hidden[0], rtol=0, atol=0)
                torch.testing.assert_close(encoder_inputs[-1][0],
                    torch.cat([x[0], cached.ring_sizes[0, :depth + 1]]), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
