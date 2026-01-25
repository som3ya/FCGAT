import torch

try:
    from torch_geometric.data import Data
except Exception:
    Data = None


def test_forward_shapes():
    """Smoke test for forward pass."""
    if Data is None:
        # torch-geometric not installed in minimal environments
        return

    from fcgat.model import FCGAT, FCGATHParams

    x = torch.randn(3, 100)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    model = FCGAT(num_classes=4, hparams=FCGATHParams())
    logits, attn_steps = model(data)

    assert logits.shape == (1, 4)
    assert isinstance(attn_steps, list)
    assert len(attn_steps) == model.hparams.set2set_steps
    assert attn_steps[-1].shape[0] == data.num_nodes
