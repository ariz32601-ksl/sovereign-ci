import torch
import torch.nn as nn

class MockEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # Sovereign-CI Auto-Patch: Upgraded input dimension from 128 to 256 to match runtime tensors
        self.linear_tf_z_i = nn.Linear(256, 22) 

    def forward(self, tf):
        tf_emb_i = self.linear_tf_z_i(tf)
        return tf_emb_i
