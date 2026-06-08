import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionMatcher(nn.Module):
    """
    CrossAttentionMatcher++ (自适应模态门控 + 对比对齐)
    -------------------------------------------------
    """

    def __init__(self, embed_dim=100, compound_struct_dim=512, target_struct_dim=147, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.compound_struct_dim = compound_struct_dim
        self.target_struct_dim = target_struct_dim

        # === 1️⃣ 投影到统一维度 ===
        self.comp_proj = nn.Sequential(
            nn.Linear(compound_struct_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.tgt_proj = nn.Sequential(
            nn.Linear(target_struct_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # === 2️⃣ 动态模态门控 (MoE-style) ===
        # 输入是 concat(embedding, struct)，输出 [α_embed, α_struct] 两个权重
        self.comp_mod_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=1)
        )
        self.tgt_mod_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=1)
        )

        # === 3️⃣ 注意力层 ===
        self.comp_att = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        self.tgt_att = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )

        # === 4️⃣ 交互特征 ===
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, 1)
        interaction_dim = embed_dim * 4 + compound_struct_dim + target_struct_dim + 1

        # === 5️⃣ 分类头 ===
        self.fc = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, c_emb, c_str, t_emb, t_str):
        # Step 1: 投影结构特征
        c_s = self.comp_proj(c_str)
        t_s = self.tgt_proj(t_str)

        # Step 2: 模态门控 (Softmax)
        g_c = self.comp_mod_gate(torch.cat([c_emb, c_s], dim=1))  # [α_embed, α_struct]
        g_t = self.tgt_mod_gate(torch.cat([t_emb, t_s], dim=1))
        α_c_emb, α_c_str = g_c[:, 0:1], g_c[:, 1:2]
        α_t_emb, α_t_str = g_t[:, 0:1], g_t[:, 1:2]

        z_c = α_c_emb * c_emb + α_c_str * c_s
        z_t = α_t_emb * t_emb + α_t_str * t_s

        # Step 3: 注意力
        a_c = self.comp_att(z_c).sigmoid()
        a_t = self.tgt_att(z_t).sigmoid()
        z_c = z_c * a_c
        z_t = z_t * a_t

        # Step 4: 交互特征
        bi = self.bilinear(z_c, z_t)
        hadamard = z_c * z_t
        absdiff = torch.abs(z_c - z_t)
        x = torch.cat([z_c, z_t, hadamard, absdiff, c_str, t_str, bi], dim=1)

        # Step 5: 分类
        logits = self.fc(x).squeeze(-1)
        return {"logits": logits, "z_c": z_c, "z_t": z_t}
