import argparse
import os
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# =========================
# 模型：RC-TransR
# 改动点：
# 1) Residual attention: alpha = 1 + beta * tanh(MLP([h_r; r; t_r])) ∈ (1-β, 1+β)
# 2) 评估时为每个候选实体重算 alpha（修复原先评估偏差）
# 3) 每步训练后对实体/关系嵌入做 L2 归一化
# =========================
class RCTransR(nn.Module):
    def __init__(self, n_entities, n_relations, dim, attn_hidden_dim=None, attn_beta=0.5, attn_dropout=0.0):
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.attn_beta = attn_beta

        # 用于随机初始化向量
        self.entity_emb = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)

        self.transfer = nn.Parameter(torch.randn(n_relations, dim, dim))

        if attn_hidden_dim is None or attn_hidden_dim <= 0:
            attn_hidden_dim = max(16, dim // 2)
        self.attn_fc1 = nn.Linear(dim * 3, attn_hidden_dim)
        self.attn_fc2 = nn.Linear(attn_hidden_dim, 1)
        self.attn_do = nn.Dropout(attn_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.transfer)
        nn.init.xavier_uniform_(self.attn_fc1.weight);
        nn.init.zeros_(self.attn_fc1.bias)
        nn.init.xavier_uniform_(self.attn_fc2.weight);
        nn.init.zeros_(self.attn_fc2.bias)

    def _project(self, e, M_r):  # e: (B,d), M_r: (B,d,d)
        return torch.bmm(M_r, e.unsqueeze(2)).squeeze(2)

    def _alpha_residual(self, concat_3d):
        # concat_3d: (..., 3d)
        x = torch.tanh(self.attn_fc1(concat_3d))
        x = self.attn_do(x)
        x = self.attn_fc2(x)  # (..., 1)
        # residual scaling in (1 - beta, 1 + beta)
        return 1.0 + self.attn_beta * torch.tanh(x)

    @torch.no_grad()
    def get_triple_embeddings(self, h, r, t):
        # 随机向量化
        h_e = self.entity_emb(h)
        t_e = self.entity_emb(t)
        r_e = self.relation_emb(r)
        # 实体到关系空间投影
        M_r = self.transfer[r]
        h_p = self._project(h_e, M_r)
        t_p = self._project(t_e, M_r)
        return h_p, r_e, t_p

    def score_triple(self, h, r, t):
        h_e = self.entity_emb(h)  # (B,d)
        t_e = self.entity_emb(t)
        r_e = self.relation_emb(r)
        M_r = self.transfer[r]  # (B,d,d)
        h_p = self._project(h_e, M_r)  # (B,d)
        t_p = self._project(t_e, M_r)

        concat_vec = torch.cat([h_p, r_e, t_p], dim=1)  # (B,3d)
        alpha = self._alpha_residual(concat_vec)  # (B,1)

        diff = alpha * (h_p + r_e - t_p)
        return -torch.norm(diff, p=2, dim=1)

    def post_update_normalize(self):
        with torch.no_grad():
            self.entity_emb.weight.div_(self.entity_emb.weight.norm(p=2, dim=1, keepdim=True).clamp_min_(1e-9))
            self.relation_emb.weight.div_(self.relation_emb.weight.norm(p=2, dim=1, keepdim=True).clamp_min_(1e-9))


# =========================
# 数据加载
# =========================
def load_triples(path, sep=None):
    if sep is None:
        for s in [',', '\t', ';', '|']:
            try:
                df = pd.read_csv(path, sep=s)
                if set(['Head', 'Relation', 'Tail']).issubset(df.columns):
                    return df[['Head', 'Relation', 'Tail']]
            except Exception:
                pass
        raise RuntimeError("Couldn't read triples file. Ensure it has header Head,Relation,Tail or provide --sep.")
    else:
        df = pd.read_csv(path, sep=sep)
        if set(['Head', 'Relation', 'Tail']).issubset(df.columns):
            return df[['Head', 'Relation', 'Tail']]
        else:
            raise RuntimeError("Provided separator didn't yield Head,Relation,Tail columns.")


# =========================
# Dataset 与 collate（负采样限定到训练实体）
# =========================
class KGDataset(Dataset):
    def __init__(self, triples, n_entities, allowed_entities=None, n_neg=1):
        self.triples = triples
        self.n_entities = n_entities
        self.allowed = list(range(n_entities)) if allowed_entities is None else list(allowed_entities)
        self.n_neg = n_neg

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        neg_triples = []
        for _ in range(self.n_neg):
            if random.random() < 0.5:
                neg_h = random.choice(self.allowed)
                neg_triples.append((neg_h, r, t))
            else:
                neg_t = random.choice(self.allowed)
                neg_triples.append((h, r, neg_t))
        return (h, r, t), neg_triples


def collate_fn(batch):
    pos = [b[0] for b in batch]
    neg = [trip for b in batch for trip in b[1]]
    return torch.LongTensor(pos), torch.LongTensor(neg)


# =========================
# 获取 embedding matrix
# =========================
def get_entity_embedding_matrix(model):
    return model.entity_emb.weight


def get_relation_embedding_matrix(model):
    return model.relation_emb.weight


# =========================
# 评分所有候选实体（修复版：为每个候选重算 alpha）
# 返回 scores_head (B,E), scores_tail (B,E)
# =========================
@torch.no_grad()
def score_candidates_transr_like(model, h_batch, r_batch, t_batch, device):
    ent_mat = get_entity_embedding_matrix(model).to(device)  # (E,d)
    rel_mat = get_relation_embedding_matrix(model).to(device)  # (R,d)
    E = ent_mat.size(0)
    B = h_batch.size(0)
    d = ent_mat.size(1)

    r_e = rel_mat[r_batch]  # (B,d)
    M_r = model.transfer[r_batch]  # (B,d,d)

    # 投影所有实体到关系空间：E_proj[b] = ent_mat @ M_r[b]^T
    E_exp = ent_mat.unsqueeze(0).expand(B, E, d)  # (B,E,d)
    E_proj = torch.bmm(E_exp, M_r.transpose(1, 2))  # (B,E,d)

    # 特定 h,t 的投影
    h_e = ent_mat[h_batch]  # (B,d)
    t_e = ent_mat[t_batch]  # (B,d)
    h_p = torch.bmm(h_e.unsqueeze(1), M_r.transpose(1, 2)).squeeze(1)  # (B,d)
    t_p = torch.bmm(t_e.unsqueeze(1), M_r.transpose(1, 2)).squeeze(1)  # (B,d)

    # ------------ 头预测：alpha(h'_p, r, t_p) ------------
    r_e_head = r_e.unsqueeze(1).expand(B, E, d)  # (B,E,d)
    t_p_head = t_p.unsqueeze(1).expand(B, E, d)  # (B,E,d)
    concat_head = torch.cat([E_proj, r_e_head, t_p_head], dim=2)  # (B,E,3d)
    head_in = concat_head.reshape(B * E, 3 * d)
    alpha_head = 1.0 + model.attn_beta * torch.tanh(
        model.attn_fc2(model.attn_do(torch.tanh(model.attn_fc1(head_in))))
    )  # (B*E,1)
    alpha_head = alpha_head.view(B, E, 1)
    diff_head = alpha_head * (E_proj + r_e_head - t_p_head)  # (B,E,d)
    scores_head = -torch.norm(diff_head, p=2, dim=2)  # (B,E)

    # ------------ 尾预测：alpha(h_p, r, t'_p) ------------
    h_p_tail = h_p.unsqueeze(1).expand(B, E, d)  # (B,E,d)
    r_e_tail = r_e.unsqueeze(1).expand(B, E, d)  # (B,E,d)
    concat_tail = torch.cat([h_p_tail, r_e_tail, E_proj], dim=2)  # (B,E,3d)
    tail_in = concat_tail.reshape(B * E, 3 * d)
    alpha_tail = 1.0 + model.attn_beta * torch.tanh(
        model.attn_fc2(model.attn_do(torch.tanh(model.attn_fc1(tail_in))))
    )  # (B*E,1)
    alpha_tail = alpha_tail.view(B, E, 1)
    diff_tail = alpha_tail * (h_p_tail + r_e_tail - E_proj)  # (B,E,d)
    scores_tail = -torch.norm(diff_tail, p=2, dim=2)  # (B,E)

    return scores_head, scores_tail


# =========================
# Filtered Evaluation
# =========================
@torch.no_grad()
def evaluate_filtered(model, eval_triples, train_triples_set,
                      n_entities, device, batch_size=64):
    model.eval()

    head_dict = {}
    tail_dict = {}
    for (hh, rr, tt) in train_triples_set:
        head_dict.setdefault((int(rr), int(tt)), set()).add(int(hh))
        tail_dict.setdefault((int(hh), int(rr)), set()).add(int(tt))

    ranks_head, ranks_tail = [], []
    hitsks = [1, 3, 10, 50, 100]
    N = len(eval_triples)

    for start in range(0, N, batch_size):
        batch = eval_triples[start:start + batch_size]
        B = len(batch)
        h_batch = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
        r_batch = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
        t_batch = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)

        true_scores = model.score_triple(h_batch, r_batch, t_batch).detach()

        scores_head, scores_tail = score_candidates_transr_like(model, h_batch, r_batch, t_batch, device)

        scores_head = scores_head.detach().cpu().numpy()
        scores_tail = scores_tail.detach().cpu().numpy()
        true_scores_cpu = true_scores.detach().cpu().numpy()

        for i in range(B):
            h_i = int(h_batch[i].cpu().item())
            r_i = int(r_batch[i].cpu().item())
            t_i = int(t_batch[i].cpu().item())

            sc = scores_head[i]
            mask_heads = head_dict.get((r_i, t_i), set())
            if h_i in mask_heads:
                mask_heads = mask_heads - {h_i}
            if mask_heads:
                sc[list(mask_heads)] = -1e9
            true_sc = float(true_scores_cpu[i])
            rank_h = int((sc > true_sc).sum()) + 1
            ranks_head.append(rank_h)

            sc2 = scores_tail[i]
            mask_tails = tail_dict.get((h_i, r_i), set())
            if t_i in mask_tails:
                mask_tails = mask_tails - {t_i}
            if mask_tails:
                sc2[list(mask_tails)] = -1e9
            rank_t = int((sc2 > true_sc).sum()) + 1
            ranks_tail.append(rank_t)

    ranks_head = np.array(ranks_head);
    ranks_tail = np.array(ranks_tail)
    if len(ranks_head) == 0 or len(ranks_tail) == 0:
        zeros = {k: 0.0 for k in hitsks}
        return 0.0, zeros, zeros, zeros

    mrr = float(np.mean(1.0 / np.concatenate([ranks_head, ranks_tail])))
    hits_head = {k: float((ranks_head <= k).mean()) for k in hitsks}
    hits_tail = {k: float((ranks_tail <= k).mean()) for k in hitsks}
    hits_avg = {k: (hits_head[k] + hits_tail[k]) / 2.0 for k in hitsks}
    return mrr, hits_head, hits_tail, hits_avg


# =========================
# 训练单折
# =========================
def train_one_fold(model, train_dataset, valid_triples, train_triples_set,
                   n_entities, args, device):
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MarginRankingLoss(margin=args.margin)

    best_mrr = -1.0
    best_hits_head = best_hits_tail = best_hits_avg = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        total_loss = 0.0
        for pos, neg in pbar:
            pos = pos.to(device);
            neg = neg.to(device)
            optimizer.zero_grad()
            pos_score = model.score_triple(pos[:, 0], pos[:, 1], pos[:, 2])
            neg_score = model.score_triple(neg[:, 0], neg[:, 1], neg[:, 2])
            y = torch.ones_like(pos_score, device=device)
            loss = criterion(pos_score, neg_score, y)
            loss.backward()
            optimizer.step()

            # 归一化（稳定 Trans 系列）
            model.post_update_normalize()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_loss = total_loss / max(1, len(loader))

        if (epoch % args.eval_every == 0) or (epoch == args.epochs):
            mrr, hits_head, hits_tail, hits_avg = evaluate_filtered(
                model, valid_triples, train_triples_set, n_entities, device,
                batch_size=args.eval_batch_size
            )
            print(f"  [Eval] Epoch {epoch} loss={avg_loss:.4f} ValMRR={mrr:.4f} "
                  f"Hits@1(avg)={hits_avg[1]:.4f} Hits@3(avg)={hits_avg[3]:.4f} Hits@10(avg)={hits_avg[10]:.4f}")
            if mrr > best_mrr:
                best_mrr = mrr
                best_hits_head = hits_head.copy()
                best_hits_tail = hits_tail.copy()
                best_hits_avg = hits_avg.copy()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_mrr, best_hits_head, best_hits_tail, best_hits_avg, best_state


# =========================
# 全量训练
# =========================
def train_on_full(model, all_triples, args, device):
    full_dataset = KGDataset(all_triples, args.n_entities, allowed_entities=None, n_neg=args.n_neg)
    loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MarginRankingLoss(margin=args.margin)

    model.train()
    for epoch in range(1, args.final_epochs + 1):
        pbar = tqdm(loader, desc=f"[Full Train] Epoch {epoch}/{args.final_epochs}", leave=False)
        total_loss = 0.0
        for pos, neg in pbar:
            pos = pos.to(device);
            neg = neg.to(device)
            optimizer.zero_grad()
            pos_score = model.score_triple(pos[:, 0], pos[:, 1], pos[:, 2])
            neg_score = model.score_triple(neg[:, 0], neg[:, 1], neg[:, 2])
            y = torch.ones_like(pos_score, device=device)
            loss = criterion(pos_score, neg_score, y)
            loss.backward()
            optimizer.step()
            model.post_update_normalize()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        print(f"[Full Train] Epoch {epoch} avg_loss={total_loss / max(1, len(loader)):.4f}")
    return model.state_dict()


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='../data/standard_data/Triplets.csv')
    parser.add_argument("--sep", type=str, default=None)
    parser.add_argument("--model", type=str, default="RCTransR",
                        choices=["RCTransR"])
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--final_epochs", type=int, default=50)
    parser.add_argument("--n_neg", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--attn_hidden_dim", type=int, default=0, help="0=auto(dim//2)")
    parser.add_argument("--attn_beta", type=float, default=0.5, help="residual attention strength in (0,1)")
    parser.add_argument("--attn_dropout", type=float, default=0.1)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load triples
    df = load_triples(args.data_path, sep=args.sep)
    df['Head'] = df['Head'].astype(str)
    df['Relation'] = df['Relation'].astype(str)
    df['Tail'] = df['Tail'].astype(str)

    entities = pd.unique(df[['Head', 'Tail']].values.ravel())
    relations = pd.unique(df['Relation'].values.ravel())
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    triples = [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in df[['Head', 'Relation', 'Tail']].values]
    triples = np.array(triples, dtype=np.int64)

    n_entities = len(ent2id)
    n_relations = len(rel2id)
    args.n_entities = n_entities
    args.n_relations = n_relations
    print(f"#triples={len(triples)}, #entities={n_entities}, #relations={n_relations}")

    # K-fold CV
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_mrrs = []
    fold_hits_head = {k: [] for k in [1, 3, 10, 50, 100]}
    fold_hits_tail = {k: [] for k in [1, 3, 10, 50, 100]}
    fold_hits_avg = {k: [] for k in [1, 3, 10, 50, 100]}
    os.makedirs("./model", exist_ok=True)

    global_best_mrr = -1.0
    global_best_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(triples), 1):
        print(f"\n=== Fold {fold}/{args.n_splits} ===")
        train_triples = triples[train_idx].tolist()
        valid_triples = triples[val_idx].tolist()

        # 确保每个实体出现在训练集中
        train_entities = set([h for h, _, t in train_triples] + [t for h, _, t in train_triples])
        missing_entities = set(range(n_entities)) - train_entities
        if missing_entities:
            for tr in valid_triples[:]:
                h, r, t = tr
                if h in missing_entities or t in missing_entities:
                    train_triples.append(tr)
                    valid_triples.remove(tr)
                    train_entities.add(h);
                    train_entities.add(t)
                    missing_entities = set(range(n_entities)) - train_entities
                    if not missing_entities:
                        break

        allowed_entities = sorted(list(train_entities))
        train_dataset = KGDataset(train_triples, n_entities,
                                  allowed_entities=allowed_entities, n_neg=args.n_neg)

        attn_hid = None if args.attn_hidden_dim == 0 else args.attn_hidden_dim
        model = RCTransR(n_entities, n_relations, args.dim,
                         attn_hidden_dim=attn_hid,
                         attn_beta=args.attn_beta,
                         attn_dropout=args.attn_dropout).to(device)

        train_triples_set = set(tuple(x) for x in train_triples)
        best_mrr, best_hits_head, best_hits_tail, best_hits_avg, best_state = train_one_fold(
            model, train_dataset, valid_triples, train_triples_set, n_entities, args, device
        )

        fold_mrrs.append(best_mrr)
        for k in fold_hits_avg.keys():
            fold_hits_head[k].append(best_hits_head[k] if best_hits_head is not None else 0.0)
            fold_hits_tail[k].append(best_hits_tail[k] if best_hits_tail is not None else 0.0)
            fold_hits_avg[k].append(best_hits_avg[k] if best_hits_avg is not None else 0.0)

        fold_path = f"./model/best_rctransr_fold{fold}_model.pth"
        if best_state is not None:
            torch.save(best_state, fold_path)
            print(f"Saved fold model: {fold_path}")

        if best_mrr > global_best_mrr and best_state is not None:
            global_best_mrr = best_mrr
            global_best_state = best_state.copy()

    # 聚合并打印 CV 结果
    print("\n==== Cross-Validation Results ====")
    mean_mrr = float(np.mean(fold_mrrs));
    std_mrr = float(np.std(fold_mrrs))
    print(f"MRR: {mean_mrr:.6f} ± {std_mrr:.6f}")
    for k in [1, 3, 10, 50, 100]:
        mean_h = float(np.mean(fold_hits_avg[k]));
        std_h = float(np.std(fold_hits_avg[k]))
        print(f"Hits@{k} (avg head+tail): {mean_h:.6f} ± {std_h:.6f}")

    # ---------- 全量训练 ----------
    print("\n==== Training Final Model on Full Data ====")
    attn_hid = None if args.attn_hidden_dim == 0 else args.attn_hidden_dim
    final_model = RCTransR(n_entities, n_relations, args.dim,
                           attn_hidden_dim=attn_hid,
                           attn_beta=args.attn_beta,
                           attn_dropout=args.attn_dropout).to(device)
    if global_best_state is not None:
        try:
            final_model.load_state_dict(global_best_state)
            print(f"Initialized final model from best CV fold (MRR={global_best_mrr:.4f})")
        except Exception as e:
            print(f"Warning: could not load global best state into final model: {e}")

    full_state = train_on_full(final_model, triples.tolist(), args, device)
    full_path = f"./model/best_rctransr_full.pth"
    torch.save(full_state, full_path)
    print(f"Saved final model (full-data): {full_path}")


if __name__ == "__main__":
    main()
