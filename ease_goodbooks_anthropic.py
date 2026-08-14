"""
EASE (Embarrassingly Shallow Autoencoder) recommender on goodbooks-10k.

EASE (Steck, 2019) is a linear item-item model with a closed-form solution
(no gradient descent / training loop needed) that is a very strong baseline
for implicit-feedback recommendation. It learns an item-item weight matrix
B such that predicted scores S = X @ B reconstruct the interaction matrix X,
subject to the constraint diag(B) = 0 (an item can't recommend itself).

Evaluated with masked recall@20: a subset of each user's interactions are
held out; the model must recover them in its top-20 ranked (unseen) items.

Usage:
    python ease_goodbooks.py

Requires: numpy, scipy, pandas
    pip install numpy scipy pandas
"""

import os
import random
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

# ----------------------------- Config ------------------------------------

SEED = 42
DATA_DIR = "data"
RATINGS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
VAL_HOLDOUT_PER_USER = 5
TOP_K = 20

# EASE regularization strength(s) to try; the one with the best validation
# recall@20 is kept. Larger goodbooks-10k-scale datasets typically want
# something in the low hundreds.
LAMBDA_GRID = [100.0, 300.0, 700.0]

random.seed(SEED)
np.random.seed(SEED)


# ----------------------------- Data ---------------------------------------

def download_data() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RATINGS_PATH):
        print(f"Downloading ratings.csv to {RATINGS_PATH} ...")
        urllib.request.urlretrieve(RATINGS_URL, RATINGS_PATH)
    df = pd.read_csv(RATINGS_PATH)
    return df[["user_id", "book_id"]].drop_duplicates()


def preprocess(df: pd.DataFrame):
    """Filter sparse users/items, build index maps, split each user's
    interactions into train / held-out validation sets."""
    rng = random.Random(SEED)

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

    return train_user_items, val_user_items, user2idx, item2idx


def build_sparse_matrix(user_items: dict, num_users: int, num_items: int) -> sparse.csr_matrix:
    rows, cols = [], []
    for u, items in user_items.items():
        rows.extend([u] * len(items))
        cols.extend(items)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))


# ----------------------------- EASE ----------------------------------------

def fit_ease(X: sparse.csr_matrix, lambda_reg: float) -> np.ndarray:
    """Closed-form EASE solution. Returns dense item-item weight matrix B
    (num_items x num_items) with a zero diagonal."""
    G = (X.T @ X).toarray().astype(np.float64)
    n_items = G.shape[0]
    G[np.diag_indices(n_items)] += lambda_reg

    c, low = cho_factor(G)
    P = cho_solve((c, low), np.eye(n_items))

    B = P / (-np.diag(P))
    np.fill_diagonal(B, 0.0)
    return B


def recall_at_k(X_train: sparse.csr_matrix, B: np.ndarray, val_user_items: dict,
                 train_user_items: dict, k: int = TOP_K) -> float:
    val_users = [u for u, items in val_user_items.items() if items]
    if not val_users:
        return 0.0

    recalls = []
    batch_size = 512
    for start in range(0, len(val_users), batch_size):
        batch_users = val_users[start:start + batch_size]
        scores = X_train[batch_users].dot(B)  # (batch, num_items), dense ndarray

        for row, u in enumerate(batch_users):
            seen = train_user_items.get(u, [])
            if seen:
                scores[row, seen] = -np.inf

        topk = np.argpartition(-scores, k, axis=1)[:, :k]
        for row, u in enumerate(batch_users):
            relevant = set(val_user_items[u])
            hits = len(relevant.intersection(topk[row].tolist()))
            recalls.append(hits / min(k, len(relevant)))

    return float(np.mean(recalls))


# ----------------------------- Main -----------------------------------------

def main():
    df = download_data()
    train_user_items, val_user_items, user2idx, item2idx = preprocess(df)
    num_users, num_items = len(user2idx), len(item2idx)
    print(f"Users: {num_users}, Items: {num_items}")

    X_train = build_sparse_matrix(train_user_items, num_users, num_items)
    print(f"Train interactions: {X_train.nnz}")

    best_recall, best_lambda, best_B = -1.0, None, None
    for lam in LAMBDA_GRID:
        print(f"Fitting EASE with lambda={lam} ...")
        B = fit_ease(X_train, lam)
        recall = recall_at_k(X_train, B, val_user_items, train_user_items)
        print(f"  lambda={lam:<8} val recall@{TOP_K} = {recall:.4f}")
        if recall > best_recall:
            best_recall, best_lambda, best_B = recall, lam, B

    print(f"\nBest lambda: {best_lambda}")
    print(f"Best validation recall@{TOP_K}: {best_recall:.4f}")


if __name__ == "__main__":
    main()
