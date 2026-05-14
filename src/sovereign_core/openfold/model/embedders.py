import torch
import torch.nn as nn
from openfold.model.primitives import Linear
from openfold.model.embedders import (
    TemplatePairEmbedder,
    TemplatePointwiseAttention,
)


class TemplateEmbedding(nn.Module):
    """
    Embeds template features.

    Implements Algorithm 18.
    """

    def __init__(self, c, **kwargs):
        super(TemplateEmbedding, self).__init__()

        self.c = c

        self.template_pair_embedder = TemplatePairEmbedder(
            self.c.template
        )
        self.template_pointwise_attention = TemplatePointwiseAttention(
            c=self.c.template,
        )

        # The input feature dimension from the traceback is 256.
        # This layer and others processing the same tensor must be updated
        # from the incorrect config value to 256.
        self.linear_tf_z_i = Linear(
            256,
            self.c.pair,
            initializer="relu",
        )
        self.linear_tf_z_j = Linear(
            256,
            self.c.pair,
            initializer="relu",
        )
        self.linear_tf_m = Linear(
            256,
            self.c.msa,
            initializer="relu",
        )
        self.linear_t_p_i = Linear(
            self.c.template.inf,
            self.c.pair,
            initializer="relu",
        )
        self.linear_t_p_j = Linear(
            self.c.template.inf,
            self.c.pair,
            initializer="relu",
        )

    def forward(self, batch):
        # [*, N_res, N_res, c_t]
        t = batch["template_pair_feat"]
        t = self.template_pair_embedder(t)

        # [*, N_templ, N_res, c_m]
        q = batch["template_pseudo_beta"]
        q = q.view(
            t.shape[:-3] + (-1, t.shape[-2], t.shape[-2], q.shape[-1])
        )

        # [*, N_templ, N_res, N_res, c_z]
        t = self.template_pointwise_attention(t, q)
        t = t.sum(dim=-4)

        # [*, N_res, c_tf]
        tf = batch["target_feat"]

        # [*, N_res, 1, c_z]
        tf_emb_i = self.linear_tf_z_i(tf).unsqueeze(-2)

        # [*, 1, N_res, c_z]
        tf_emb_j = self.linear_tf_z_j(tf).unsqueeze(-3)

        # [*, N_res, c_m]
        tf_emb_m = self.linear_tf_m(tf)

        # [*, N_res, 1, c_z]
        t_p_i = self.linear_t_p_i(t).unsqueeze(-2)

        # [*, 1, N_res, c_z]
        t_p_j = self.linear_t_p_j(t).unsqueeze(-3)

        return t_p_i, t_p_j, tf_emb_i, tf_emb_j, tf_emb_m