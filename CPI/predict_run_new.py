# predict_run_new.py

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from predictor_model_new import CrossAttentionMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler_c = StandardScaler()
scaler_t = StandardScaler()

# ========== 参数 ==========
model_mode = 2  # 0: CV only, 1: train only, 2: both

EMB_PATH = "../data/results/entity_embeddings.npy"
ENTITY2ID = "../data/results/entity2id.json"
COMP_NAMES = "../data/standard_data/all_component_names.csv"
TGT_NAMES = "../data/standard_data/all_target_names.csv"
COMP_STRUCT = "../data/structural_data/fingerprints_512.csv"
TGT_STRUCT = "../data/structural_data/protein_CTD_descriptors.csv"
PAIR_LABELS = "../data/standard_data/positive_negative_pairs.csv"
MODEL_SAVE_PATH = "../data/results/final_predictor_rctransr_AndCTDAndFingerprints.pth"

# 训练超参
EPOCHS = 60
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 1.5
ALIGN_LAMBDA = 0.7
COS_TEMPERATURE = 6.0


# ========== 加载数据 ==========
def load_data():
    emb = np.load(EMB_PATH)
    with open(ENTITY2ID, encoding='utf-8') as f:
        entity2id = json.load(f)

    # 成分和靶点名称
    raw_comp_names = pd.read_csv(COMP_NAMES, header=0).iloc[:, 0].dropna().astype(str).tolist()
    raw_tgt_names = pd.read_csv(TGT_NAMES, header=0).iloc[:, 0].dropna().astype(str).tolist()

    # 仅保留在 entity2id 中存在的成分/靶点名称
    comp_names = [name for name in raw_comp_names if name in entity2id]
    tgt_names = [name for name in raw_tgt_names if name in entity2id]

    # 成分结构向量（指纹）
    comp_fp_raw = pd.read_csv(COMP_STRUCT, usecols=["molecule_name", "Morgan_Fingerprint"])
    comp_fp_raw["Morgan_Fingerprint"] = comp_fp_raw["Morgan_Fingerprint"].apply(
        lambda x: np.array(list(x.strip()), dtype=np.float32)
    )
    comp_fp_dict = dict(zip(comp_fp_raw["molecule_name"], comp_fp_raw["Morgan_Fingerprint"]))

    # 过滤掉结构中找不到的成分
    comp_names = [name for name in comp_names if name in comp_fp_dict]
    comp_struct = np.stack([comp_fp_dict[name] for name in comp_names])

    # 靶点结构向量（CTD）
    tgt_df = pd.read_csv(TGT_STRUCT)
    tgt_df.set_index(tgt_df.columns[0], inplace=True)

    tgt_names = [name for name in tgt_names if name in tgt_df.index]
    tgt_struct = tgt_df.loc[tgt_names].astype(np.float32).to_numpy()

    # —— 重要：全量标准化（避免折间泄露：下面在 Dataset 里仅做 transform）
    comp_struct_std = scaler_c.fit_transform(comp_struct)
    tgt_struct_std = scaler_t.fit_transform(tgt_struct)

    return emb, entity2id, comp_names, tgt_names, comp_struct_std, tgt_struct_std


# ========== 数据集 ==========
class CPTRDataset(Dataset):
    def __init__(self, df, emb, e2id, comp_struct_std, tgt_struct_std, comp_names, tgt_names):
        self.data = df.reset_index(drop=True)
        self.emb = emb
        self.e2id = e2id
        self.comp_struct = comp_struct_std
        self.tgt_struct = tgt_struct_std
        self.comp_names = comp_names
        self.tgt_names = tgt_names

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        comp_name = row["Component"]
        target_name = row["Target"]

        c_id = self.e2id[comp_name]
        t_id = self.e2id[target_name]
        c_idx = self.comp_names.index(comp_name)
        t_idx = self.tgt_names.index(target_name)

        c_emb = torch.tensor(self.emb[c_id], dtype=torch.float32)
        c_str = torch.tensor(self.comp_struct[c_idx], dtype=torch.float32)
        t_emb = torch.tensor(self.emb[t_id], dtype=torch.float32)
        t_str = torch.tensor(self.tgt_struct[t_idx], dtype=torch.float32)
        label = torch.tensor(float(row["Label"]), dtype=torch.float32)
        return c_emb, c_str, t_emb, t_str, label


# ========== 评估（含阈值扫描）==========
def evaluate(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for c, c_s, t, t_s, y in loader:
            out = model(c.to(device), c_s.to(device), t.to(device), t_s.to(device))
            logits = out["logits"].detach().cpu()
            prob = torch.sigmoid(logits)
            probs.extend(prob.numpy())
            labels.extend(y.numpy())

    probs = np.array(probs)
    labels = np.array(labels)

    # 阈值扫描 -> F1 最优阈值
    best_f1, best_thr, best_prec, best_rec = 0, 0.5, 0, 0
    for thr in np.arange(0.1, 0.91, 0.01):
        pred_bin = (probs > thr).astype(int)
        f1 = f1_score(labels, pred_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_prec = precision_score(labels, pred_bin, zero_division=0)
            best_rec = recall_score(labels, pred_bin, zero_division=0)

    # PR-AUC（注意 sklearn 返回的是 (precision, recall, thresholds)）
    precision_arr, recall_arr, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall_arr, precision_arr)
    pred_bin = (probs > best_thr).astype(int)

    result = {
        'acc': accuracy_score(labels, pred_bin),
        'f1': best_f1,
        'auc': roc_auc_score(labels, probs),
        'pr_auc': pr_auc,
        'precision': best_prec,
        'recall': best_rec,
        'best_thr': best_thr
    }

    print(f"最佳阈值: {best_thr:.2f}  Precision: {best_prec:.3f}  Recall: {best_rec:.3f}  F1: {best_f1:.3f}")
    return result


# ========== 损失函数 ==========
def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    """
    logits: (B,)
    targets: (B,)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)  # pt = sigmoid/logit 的正确概率
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def contrastive_align_loss(z_c, z_t, targets, temperature=5.0):
    """
    用余弦相似度来对齐：正样本相似度高、负样本低。
    s = cos(z_c, z_t) * temperature  ->  BCEWithLogits(s, targets)
    """
    z_c = F.normalize(z_c, dim=1)
    z_t = F.normalize(z_t, dim=1)
    cos_sim = F.cosine_similarity(z_c, z_t)  # [-1, 1]
    logits = cos_sim * temperature
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


# ========== 训练 ==========
def train_model(train_data, val_data, emb, e2id, comp_struct, tgt_struct, save_path=None):
    # 构建数据
    train_set = CPTRDataset(train_data, emb, e2id, comp_struct, tgt_struct, comp_names, tgt_names)
    val_set = CPTRDataset(val_data, emb, e2id, comp_struct, tgt_struct, comp_names, tgt_names)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # 模型 & 优化器
    model = CrossAttentionMatcher().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for c, c_s, t, t_s, y in train_loader:
            out = model(c.to(device), c_s.to(device), t.to(device), t_s.to(device))
            logits = out["logits"]
            z_c = out["z_c"]
            z_t = out["z_t"]
            y = y.to(device)

            # 主损失：Focal
            loss_cls = focal_loss_with_logits(logits, y, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
            # 辅损失：对比对齐
            loss_align = contrastive_align_loss(z_c, z_t, y, temperature=COS_TEMPERATURE)

            loss = loss_cls + ALIGN_LAMBDA * loss_align

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}  Loss: {total / max(1, len(train_loader)):.4f}")

    # 评估与保存
    if save_path:
        torch.save(model.state_dict(), save_path)
    return model, evaluate(model, val_loader)


# ========== 主函数 ==========
if __name__ == "__main__":
    from sklearn.metrics import f1_score  # 局部导入，避免顶部冗长

    emb, entity2id, comp_names, tgt_names, comp_struct, tgt_struct = load_data()
    pair_df = pd.read_csv(PAIR_LABELS)

    # 过滤：必须能在嵌入和结构里找到
    valid_mask = pair_df["Component"].isin(entity2id) & pair_df["Target"].isin(entity2id)
    pair_df = pair_df[valid_mask].reset_index(drop=True)
    pair_df = pair_df[pair_df["Component"].isin(comp_names) & pair_df["Target"].isin(tgt_names)].reset_index(drop=True)

    if model_mode in [0, 2]:
        print("=== Cross Validation ===")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(pair_df)):
            print(f"\nFold {fold + 1}")
            model, result = train_model(pair_df.iloc[train_idx], pair_df.iloc[val_idx],
                                        emb, entity2id, comp_struct, tgt_struct,
                                        save_path=None)
            print(result)
            scores.append(result)

        print("\n==== 十折交叉验证平均性能（均值 ± 标准差） ====")
        df = pd.DataFrame(scores)[['acc', 'f1', 'auc', 'pr_auc', 'precision', 'recall']]
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            print(f"{col:10s}: {mean:.4f} ± {std:.4f}")

    if model_mode in [1, 2]:
        print("\n=== 训练最终模型并保存 ===")
        train_model(pair_df, pair_df, emb, entity2id, comp_struct, tgt_struct,
                    save_path=MODEL_SAVE_PATH)
        print(f"✅ 最终模型已保存至 {MODEL_SAVE_PATH}")
