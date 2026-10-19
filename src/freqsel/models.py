import torch, torch.nn as nn

class OneLayer(nn.Module):
    """1-layer attention transformer, no LayerNorm; parameter creation order = original experiments."""
    def __init__(self, p=113, d=128, h=4, d_mlp=512, n_ctx=3):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(p + 1, d) / d**0.5)
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d) / d**0.5)
        self.W_Q = nn.Parameter(torch.randn(h, d, d // h) / d**0.5)
        self.W_K = nn.Parameter(torch.randn(h, d, d // h) / d**0.5)
        self.W_V = nn.Parameter(torch.randn(h, d, d // h) / d**0.5)
        self.W_O = nn.Parameter(torch.randn(h, d // h, d) / d**0.5)
        self.W_in = nn.Parameter(torch.randn(d, d_mlp) / d**0.5)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d) / d_mlp**0.5)
        self.b_out = nn.Parameter(torch.zeros(d))
        self.W_U = nn.Parameter(torch.randn(d, p) / d**0.5)

class MLPOnly(nn.Module):
    def __init__(self, p=113, d=128, d_mlp=512):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(p + 1, d) / d**0.5)
        self.W_in = nn.Parameter(torch.randn(2 * d, d_mlp) / (2 * d)**0.5)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d) / d_mlp**0.5)
        self.b_out = nn.Parameter(torch.zeros(d))
        self.W_U = nn.Parameter(torch.randn(d, p) / d**0.5)

def init_state(seed, p=113, arch="attn"):
    torch.manual_seed(seed)
    m = OneLayer(p) if arch == "attn" else MLPOnly(p)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}
