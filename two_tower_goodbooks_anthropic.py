"""
Two-Tower recommender model on the goodbooks-10k dataset.

Trains a user-tower / item-tower embedding model with in-batch softmax
(sampled) loss and logQ popularity correction, then evaluates with
masked recall@20: for each user, some of their book interactions are
held out; the model must recover them in its top-20 ranked list
(restricted to items the user hasn't already interacted with in train).

Usage:
    python two_tower_goodbooks.py

Requires: torch, pandas, numpy
    pip install torch pandas numpy
"""

import os
import random
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ----------------------------- Config ------------------------------------

SEED = 42
DATA_DIR = "data"
RATINGS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")

MIN_USER_INTERACTIONS = 5   # drop users with fewer ratings than this
MIN_ITEM_INTERACTIONS = 5   # drop items with fewer ratings than this
VAL_HOLDOUT_PER_USER = 5    # max items held out per user for validation

EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-6
TEMPERATURE = 0.05
EPOCHS = 30
EVAL_EVERY = 2
TOP_K = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------- Data ---------------------------------------

def download_data() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RATINGS_PATH):
        print(f"Downloading ratings.csv to {RATINGS_PATH} ...")
        urllib.request.urlretrieve(RATINGS_URL, RATINGS_PATH)
    df = pd.read_csv(RATINGS_PATH)
    # goodbooks-10k ships as (user_id, book_id, rating); we use it as
    # implicit feedback (interaction happened) regardless of the rating value.
    return df[["user_id", "book_id"]].drop_duplicates()


def preprocess(df: pd.DataFrame):
    """Filter sparse users/items, build index maps, and split each user's
    interactions into train / validation (held-out) sets."""
    rng = random.Random(SEED)

    # Iteratively filter since removing items can push some users below
    # the threshold and vice versa.
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["book_id"].value_counts()
        good_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
        good_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        filtered = df[df["user_id"].isin(good_users) & df["book_id"].isin(good_items)]
        if len(filtered) == len(df):
            df = filtered
            break
        df = filtered

    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["book_id"].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {b: i for i, b in enumerate(item_ids)}

    user_to_items = defaultdict(list)
    for u, b in zip(df["user_id"], df["book_id"]):
        user_to_items[user2idx[u]].append(item2idx[b])

    train_user_items = defaultdict(list)
    val_user_items = defaultdict(list)
    for u, items in user_to_items.items():
        items = items[:]
        rng.shuffle(items)
        k = min(VAL_HOLDOUT_PER_USER, max(1, len(items) // 5))
        val_items, train_items = items[:k], items[k:]
        if not train_items:  # safety net for very small users
            train_items, val_items = val_items, []
        train_user_items[u] = train_items
        val_user_items[u] = val_items

    train_pairs = [(u, i) for u, items in train_user_items.items() for i in items]
    return train_pairs, train_user_items, val_user_items, user2idx, item2idx


class InteractionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        return u, i


# ----------------------------- Model ---------------------------------------

class Tower(nn.Module):
    """Embedding lookup + small residual MLP, L2-normalized output."""

    def __init__(self, num_ids: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_ids, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.05)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)
        x = x + self.mlp(x)
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.user_tower = Tower(num_users, emb_dim, hidden_dim)
        self.item_tower = Tower(num_items, emb_dim, hidden_dim)

    def forward(self, u_ids: torch.Tensor, i_ids: torch.Tensor):
        return self.user_tower(u_ids), self.item_tower(i_ids)


# ----------------------------- Train / Eval ---------------------------------

def train_epoch(model, loader, optimizer, device, log_q, temperature):
    model.train()
    total_loss, total_n = 0.0, 0
    for u, i in loader:
        u, i = u.to(device), i.to(device)
        u_emb, i_emb = model(u, i)
        logits = (u_emb @ i_emb.t()) / temperature
        # logQ correction: in-batch items act as sampled negatives, drawn
        # roughly proportional to their training frequency, so we subtract
        # log(prob) per item column to debias the softmax (Yi et al., 2019).
        logits = logits - log_q[i].unsqueeze(0)
        labels = torch.arange(u.size(0), device=device)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * u.size(0)
        total_n += u.size(0)
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, train_user_items, val_user_items, num_items, device, k=TOP_K, batch_size=256):
    model.eval()
    all_item_ids = torch.arange(num_items, device=device)
    item_emb = model.item_tower(all_item_ids)  # (num_items, dim)

    val_users = [u for u, items in val_user_items.items() if items]
    if not val_users:
        return 0.0

    recalls = []
    for start in range(0, len(val_users), batch_size):
        batch_users = val_users[start:start + batch_size]
        u_tensor = torch.tensor(batch_users, device=device)
        u_emb = model.user_tower(u_tensor)
        scores = u_emb @ item_emb.t()  # (batch, num_items)

        for row, u in enumerate(batch_users):
            seen = train_user_items.get(u, [])
            if seen:
                scores[row, seen] = -1e9  # mask already-seen items

        topk = torch.topk(scores, k, dim=1).indices.cpu().numpy()
        for row, u in enumerate(batch_users):
            relevant = set(val_user_items[u])
            hits = len(relevant.intersection(topk[row].tolist()))
            recalls.append(hits / min(k, len(relevant)))

    return float(np.mean(recalls))


# ----------------------------- Main -----------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = download_data()
    train_pairs, train_user_items, val_user_items, user2idx, item2idx = preprocess(df)
    num_users, num_items = len(user2idx), len(item2idx)
    print(f"Users: {num_users}, Items: {num_items}, Train pairs: {len(train_pairs)}")

    item_counts = np.ones(num_items)  # +1 smoothing so unseen items aren't log(0)
    for _, i in train_pairs:
        item_counts[i] += 1
    item_freq = item_counts / item_counts.sum()
    log_q = torch.log(torch.tensor(item_freq, dtype=torch.float32)).to(device)

    dataset = InteractionDataset(train_pairs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = TwoTowerModel(num_users, num_items, EMBED_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_recall = 0.0
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, device, log_q, TEMPERATURE)
        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            recall = evaluate(model, train_user_items, val_user_items, num_items, device)
            best_recall = max(best_recall, recall)
            print(f"epoch {epoch:2d} | loss {loss:.4f} | val recall@{TOP_K} {recall:.4f}")

    print(f"\nBest validation recall@{TOP_K}: {best_recall:.4f}")


if __name__ == "__main__":
    main()
