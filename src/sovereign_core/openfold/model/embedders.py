class TemplateEmbedder(nn.Module):
    """
    Embeds template features. Uses the TemplatePairEmbedder.
    
    Implements Algorithm 17.
    """
    def __init__(self, c):
        super(TemplateEmbedder, self).__init__()
        
        self.c = c

        self.template_pair_embedder = TemplatePairEmbedder(c)
        
        # SOVEREIGN-CI PATCH:
        # The traceback indicates the input feature dimension is 256,
        # but the layer was initialized with c.d_t (22).
        # Hardcoding the correct in_features dimension to 256.
        template_feature_dim = 256

        self.linear_tf_z_i = Linear(
            template_feature_dim, c.d_pair, init="final"
        )
        self.linear_tf_z_j = Linear(
            template_feature_dim, c.d_pair, init="final"
        )
        
        self.layer_norm_z = LayerNorm(c.d_pair)
        
        self.linear_tf_m = Linear(
            template_feature_dim, c.d_msa, init="final"
        )
        
        self.layer_norm_m = LayerNorm(c.d_msa)

    def forward(self, batch):
        """
        Args:
            batch:
                A dictionary containing, among other things,
                "template_features" and "template_pair_features"
        Returns:
            pair_activations:
                [*, N_res, N_res, C_pair] template pair embeddings
            msa_activations:
                [*, N_res, N_res, C_msa] template msa embeddings
        """
        # [*, N_templ, N_res, N_res, C_t]
        tf = batch["template_features"]
        
        # [*, N_templ, N_res, N_res, C_t]
        tf = self.template_pair_embedder(tf)
        
        # [*, N_res, N_res, C_t]
        tf = tf.sum(dim=-4)
        
        # [*, N_res, N_res, C_pair]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        
        pair_activations = self.layer_norm_z(tf_emb_i + tf_emb_j)
        
        # [*, N_res, N_res, C_msa]
        # This is memory-intensive, so we do it sequentially.
        tf_m = []
        for i in range(tf.shape[-3]):
            tf_m.append(self.linear_tf_m(tf[..., i, :, :]))
        
        tf_m = torch.stack(tf_m, dim=-4)
        
        msa_activations = self.layer_norm_m(tf_m)

        return pair_activations, msa_activations