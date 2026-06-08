# scoring.py

import json
import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler
from predictor_model_new import CrossAttentionMatcher

# ========== 路径与常量（与训练保持一致） ==========
EMB_PATH = "../../comparative experiment/results/entity_embeddings.npy"
ENTITY2ID = "../../comparative experiment/results/entity2id.json"
COMP_NAMES = "../../data/standard_data/all_component_names.csv"
TGT_NAMES = "../../data/standard_data/all_target_names.csv"
COMP_STRUCT = "../../data/structural_data/fingerprints_512.csv"
TGT_STRUCT = "../../data/structural_data/protein_CTD_descriptors.csv"
PAIR_LABELS = "../../data/standard_data/positive_negative_pairs.csv"
MODEL_PATH = "./final_predictor_rctransr_AndCTDAndFingerprints_512.pth"

TOP_N = 130000  # 输出的Top候选数


# ========== 数据加载部分 ==========
def load_data():
    """
    加载嵌入、成分/靶点名称与结构特征。
    - 以 fingerprints_512.csv 为准过滤成分（缺失指纹的剔除）
    - 以 CTD 描述符表为准过滤靶点（缺失CTD的剔除）
    - 结构特征加载后立即标准化（与训练保持一致）
    """
    # --- 嵌入与实体映射（与训练一致） ---
    emb = np.load(EMB_PATH)
    with open(ENTITY2ID, "r", encoding="utf-8") as f:
        entity2id = json.load(f)

    # --- 成分与靶点名称（只保留 entity2id 中存在的） ---
    raw_comp_names = (
        pd.read_csv(COMP_NAMES, header=0).iloc[:, 0].dropna().astype(str).tolist()
    )
    raw_tgt_names = (
        pd.read_csv(TGT_NAMES, header=0).iloc[:, 0].dropna().astype(str).tolist()
    )
    comp_names = [n for n in raw_comp_names if n in entity2id]
    tgt_names = [n for n in raw_tgt_names if n in entity2id]

    # --- 成分指纹（按文件存在性过滤） ---
    comp_fp_raw = pd.read_csv(
        COMP_STRUCT, usecols=["molecule_name", "Morgan_Fingerprint"]
    )
    comp_fp_raw["Morgan_Fingerprint"] = comp_fp_raw["Morgan_Fingerprint"].apply(
        lambda x: np.array(list(str(x).strip()), dtype=np.float32)
    )
    comp_fp_dict = dict(
        zip(
            comp_fp_raw["molecule_name"].astype(str),
            comp_fp_raw["Morgan_Fingerprint"],
        )
    )
    filtered_comp_names = [n for n in comp_names if n in comp_fp_dict]
    removed = set(comp_names) - set(filtered_comp_names)
    if removed:
        print(
            f"[Warning] {len(removed)} components missing fingerprints and will be ignored, "
            f"e.g. {list(removed)[:5]}"
        )
    comp_names = filtered_comp_names
    comp_struct = np.stack([comp_fp_dict[name] for name in comp_names])

    # --- 靶点CTD（按索引存在性过滤） ---
    tgt_df = pd.read_csv(TGT_STRUCT)
    tgt_df.set_index(tgt_df.columns[0], inplace=True)
    tgt_names = [n for n in tgt_names if n in tgt_df.index]
    tgt_struct = tgt_df.loc[tgt_names].astype(np.float32).to_numpy()

    # --- 结构特征标准化（与训练一致） ---
    comp_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    comp_struct = comp_scaler.fit_transform(comp_struct)
    tgt_struct = tgt_scaler.fit_transform(tgt_struct)

    return emb, entity2id, comp_names, tgt_names, comp_struct, tgt_struct


# ========== 过滤已知 pairs ==========
def filter_known_pairs(comp_names, tgt_names):
    """
    从所有成分-靶点组合中剔除 positive_negative_pairs.csv 中已出现过的已知对
    """
    df = pd.read_csv(PAIR_LABELS)
    known_pairs = set(zip(df["Component"], df["Target"]))
    pairs = [
        (c, t)
        for c in comp_names
        for t in tgt_names
        if (c, t) not in known_pairs
    ]
    print(f"Total candidate pairs after filtering known ones: {len(pairs)}")
    return pairs


# ========== 打分函数 ==========
def score_pairs(
        pairs,
        comp_names,
        tgt_names,
        comp_struct,
        tgt_struct,
        emb,
        e2id,
        model_path=MODEL_PATH,
        top_n=TOP_N,
):
    """
    使用已训练好的 CrossAttentionMatcher 对未知成分-靶点对进行打分。
    同时输出 logit（原始分数）与 sigmoid(score)（概率）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = emb.shape[1]
    comp_dim = comp_struct.shape[1]
    tgt_dim = tgt_struct.shape[1]

    model = CrossAttentionMatcher(
        embed_dim=embed_dim,
        compound_struct_dim=comp_dim,
        target_struct_dim=tgt_dim,
        hidden_dim=256,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scores = []
    with torch.no_grad():
        for comp, tgt in pairs:
            if comp not in e2id or tgt not in e2id:
                continue
            try:
                c_idx = comp_names.index(comp)
                t_idx = tgt_names.index(tgt)
            except ValueError:
                continue

            c_emb = torch.tensor(emb[e2id[comp]], dtype=torch.float32, device=device).unsqueeze(0)
            t_emb = torch.tensor(emb[e2id[tgt]], dtype=torch.float32, device=device).unsqueeze(0)
            c_str = torch.tensor(comp_struct[c_idx], dtype=torch.float32, device=device).unsqueeze(0)
            t_str = torch.tensor(tgt_struct[t_idx], dtype=torch.float32, device=device).unsqueeze(0)

            output = model(c_emb, c_str, t_emb, t_str)  # output 是 dict
            logit = output["logits"]  # 取出 logits
            prob = torch.sigmoid(logit)
            scores.append((comp, tgt, logit.item(), prob.item()))

    # 排序与输出
    top_scores = sorted(scores, key=lambda x: -x[2])[:top_n]

    print("\nTop 20 predicted pairs:")
    for comp, tgt, logit, score in top_scores[:20]:
        print(f"{comp} - {tgt}: logit={logit:.4f}, score={score:.4f}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(top_scores, columns=["Compound", "Target", "Logit", "Score"]).to_csv(
        "results/top_predicted_pairs_new.csv", index=False
    )
    print("✅ Top predicted pairs (with both Logit & Score) saved to results/top_predicted_pairs_new.csv")


# ========== 主入口 ==========
if __name__ == "__main__":
    emb, e2id, comp_names, tgt_names, comp_struct, tgt_struct = load_data()
    pairs = filter_known_pairs(comp_names, tgt_names)
    score_pairs(pairs, comp_names, tgt_names, comp_struct, tgt_struct, emb, e2id)
