#!/usr/bin/env python
"""
PyTorch two-tower retrieval model for GoodBooks-10k with masked Recall@20.

This script:
  - Loads GoodBooks-10k ratings.csv
  - Filters cold users/items
  - Holds out one masked interaction per user for validation
  - Trains a two-tower model using in-batch sampled softmax
  - Evaluates full-catalog masked Recall@20, excluding training items

Example:
    python goodbooks_two_tower_pytorch.py --ratings ratings.csv
"""

import argparse
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.cuda.amp import GradScaler, autocast
except Exception:
    GradScaler = None
    autocast = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch two-tower GoodBooks-10k model targeting masked Recall@20 > 0.20"
    )

    parser.add_argument("--ratings", type=str, default="ratings.csv",
                        help="Path to GoodBooks-10k ratings.csv")
    parser.add_argument("--min-rating", type=float, default=3.0,
                        help="Keep ratings >= this value. Use 0 to keep all ratings.")
    parser.add_argument("--min-user-interactions", type=int, default=10,
                        help="Minimum interactions per user.")
    parser.add_argument("--min-item-interactions", type=int, default=10,
                        help="Minimum interactions per book.")

    parser.add_argument("--dim", type=int, default=128,
                        help="Embedding dimension.")
    parser.add_argument("--epochs", type=int, default=12,
                        help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Training batch size. Reduce to 2048 if you get OOM.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay.")
    parser.add_argument("--temperature", type=float, default=0.10,
                        help="Temperature for cosine similarity logits.")
    parser.add_argument("--pop-bias-scale", type=float, default=2.0,
                        help="Initial item bias scale from log popularity. Set 0 to disable.")

    parser.add_argument("--k", type=int, default=20,
                        help="Recall cutoff.")
    parser.add_argument("--target-recall", type=float, default=0.20,
                        help="Stop training once validation recall reaches this value.")
    parser.add_argument("--eval-batch", type=int, default=2048,
                        help="Batch size for validation evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_filter_ratings(args):
    """
    Load GoodBooks ratings and filter cold-start users/items.

    GoodBooks ratings.csv has columns:
        user_id, book_id, rating
    """
    df = pd.read_csv(args.ratings)

    if args.min_rating > 0 and "rating" in df.columns:
        df = df[df["rating"] >= args.min_rating]

    # Implicit feedback: only user-item interaction matters.
    df = df[["user_id", "book_id"]].drop_duplicates()

    # Iteratively filter users and items until stable.
    while True:
        before = len(df)

        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= args.min_user_interactions].index)]

        item_counts = df["book_id"].value_counts()
        df = df[df["book_id"].isin(item_counts[item_counts >= args.min_item_interactions].index)]

        if len(df) == before:
            break

    df = df.reset_index(drop=True)

    # Convert original IDs to contiguous integer IDs.
    df["user_idx"], _ = pd.factorize(df["user_id"])
    df["book_idx"], _ = pd.factorize(df["book_id"])

    if len(df) == 0:
        raise ValueError("No interactions left after filtering. Try weaker filters.")

    num_users = int(df["user_idx"].max()) + 1
    num_books = int(df["book_idx"].max()) + 1

    return df, num_users, num_books


def masked_split(df: pd.DataFrame, seed: int):
    """
    Hold out one random interaction per user for validation.
    The remaining interactions are used for training.
    """
    val_idx = df.groupby("user_idx").sample(1, random_state=seed).index

    val = df.loc[val_idx, ["user_idx", "book_idx"]].to_numpy(np.int64)
    train = df.drop(index=val_idx)[["user_idx", "book_idx"]].reset_index(drop=True)

    return train, val


def build_seen_items(train_df: pd.DataFrame, num_users: int):
    """
    Build a list of training items seen by each user.
    These are excluded during masked Recall@K evaluation.
    """
    seen = [np.empty(0, dtype=np.int64) for _ in range(num_users)]

    grouped = train_df.groupby("user_idx")["book_idx"].apply(
        lambda s: s.to_numpy(np.int64)
    )

    for user_idx, items in grouped.items():
        seen[int(user_idx)] = items

    return seen


def make_popularity_bias_init(counts: np.ndarray, scale: float):
    """
    Initialize item bias with normalized log popularity.
    This gives the model a reasonable popularity prior.
    """
    if scale <= 0:
        return np.zeros(len(counts), dtype=np.float32)

    pop = np.log1p(counts.astype(np.float32))
    mx = pop.max()

    if mx > 0:
        pop = pop / mx

    return (pop * scale).astype(np.float32)


class TwoTowerModel(nn.Module):
    """
    Two-tower retrieval model.

    User tower:
        user_id -> embedding -> dense -> dense -> L2 normalize

    Item tower:
        book_id -> embedding -> linear projection -> L2 normalize

    Retrieval score:
        dot(user_vec, item_vec) / temperature + item_bias
    """

    def __init__(self, num_users: int, num_books: int, dim: int, item_bias_init=None):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, dim)
        self.user_hidden = nn.Linear(dim, dim)
        self.user_output = nn.Linear(dim, dim, bias=False)

        self.item_embedding = nn.Embedding(num_books, dim)
        self.item_projection = nn.Linear(dim, dim, bias=False)

        if item_bias_init is None:
            self.item_bias = nn.Parameter(torch.zeros(num_books))
        else:
            self.item_bias = nn.Parameter(
                torch.tensor(item_bias_init, dtype=torch.float32)
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

        nn.init.xavier_uniform_(self.user_hidden.weight)
        nn.init.zeros_(self.user_hidden.bias)

        nn.init.xavier_uniform_(self.user_output.weight)
        nn.init.xavier_uniform_(self.item_projection.weight)

    def user_tower(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.user_embedding(user_ids)
        x = F.relu(self.user_hidden(x))
        x = self.user_output(x)
        return F.normalize(x, dim=-1)

    def item_tower(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.item_embedding(item_ids)
        x = self.item_projection(x)
        return F.normalize(x, dim=-1)


def in_batch_softmax_loss(
    model: TwoTowerModel,
    users: torch.Tensor,
    items: torch.Tensor,
    item_log_probs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    In-batch sampled softmax loss.

    Each batch provides positive diagonal pairs and negative off-diagonal pairs.

    A log-popularity correction is applied to reduce popularity sampling bias:
        logits -= log(p_item)
    """
    batch_size = items.size(0)

    user_vec = model.user_tower(users) / temperature       # [B, D]
    item_vec = model.item_tower(items)                     # [B, D]
    item_bias = model.item_bias[items]                     # [B]

    # [B, B]
    logits = torch.matmul(user_vec, item_vec.t()) + item_bias.unsqueeze(0)

    # Log-Q correction for in-batch candidates.
    logits = logits - item_log_probs[items].unsqueeze(0)

    # If the same item appears multiple times in the batch,
    # do not treat off-diagonal duplicates as negatives.
    same_item = items.unsqueeze(0) == items.unsqueeze(1)
    diag = torch.eye(batch_size, dtype=torch.bool, device=items.device)
    duplicate_negative_mask = same_item & (~diag)

    logits = logits.masked_fill(duplicate_negative_mask, -1e4)

    labels = torch.arange(batch_size, device=items.device)
    loss = F.cross_entropy(logits, labels)

    return loss


@torch.no_grad()
def masked_recall_at_k(
    model: TwoTowerModel,
    val: np.ndarray,
    seen_tensors: list,
    temperature: float,
    k: int,
    eval_batch_size: int,
    device: torch.device,
) -> float:
    """
    Full-catalog masked Recall@K.

    For each validation user:
      - Score all books.
      - Mask all books seen during training.
      - Check whether the held-out book is in top-K.
    """
    model.eval()

    val_users = val[:, 0]
    val_items = val[:, 1]
    n = len(val_users)

    num_books = model.item_embedding.num_embeddings
    k_eff = min(k, num_books)

    all_items = torch.arange(num_books, device=device)

    item_emb = model.item_tower(all_items)          # [num_books, dim]
    item_bias = model.item_bias.detach()            # [num_books]

    hits = 0

    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)

        users = torch.from_numpy(val_users[start:end]).long().to(device)
        targets = torch.from_numpy(val_items[start:end]).long().to(device)

        user_emb = model.user_tower(users) / temperature

        # [batch, num_books]
        scores = torch.matmul(user_emb, item_emb.t()) + item_bias.unsqueeze(0)

        # Mask training items per user.
        for i, user_idx in enumerate(users.tolist()):
            seen = seen_tensors[user_idx]
            if seen.numel() > 0:
                scores[i].index_fill_(0, seen, float("-inf"))

        top_k = torch.topk(scores, k_eff, dim=1).indices

        hit = (top_k == targets.unsqueeze(1)).any(dim=1)
        hits += hit.sum().item()

    return hits / n


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading and filtering GoodBooks-10k ratings...")
    df, num_users, num_books = load_and_filter_ratings(args)

    print("Creating masked validation split...")
    train_df, val = masked_split(df, seed=args.seed)

    print(f"Users: {num_users:,}")
    print(f"Books: {num_books:,}")
    print(f"Train interactions: {len(train_df):,}")
    print(f"Validation users: {len(val):,}")

    # Item counts from training only.
    train_item_counts = np.bincount(
        train_df["book_idx"].to_numpy(np.int64),
        minlength=num_books,
    ).astype(np.int64)

    # Avoid zero counts.
    train_item_counts = np.maximum(train_item_counts, 1)

    # Log-probabilities for in-batch log-Q correction.
    counts_float = train_item_counts.astype(np.float64)
    item_probs = counts_float / counts_float.sum()
    item_log_probs = torch.tensor(
        np.log(np.clip(item_probs, 1e-12, None)),
        dtype=torch.float32,
        device=device,
    )

    # Seen items per user for evaluation masking.
    seen_items = build_seen_items(train_df, num_users)
    seen_tensors = [
        torch.as_tensor(arr, dtype=torch.long, device=device)
        for arr in seen_items
    ]

    # Popularity-initialized item bias.
    item_bias_init = make_popularity_bias_init(
        counts=train_item_counts,
        scale=args.pop_bias_scale,
    )

    model = TwoTowerModel(
        num_users=num_users,
        num_books=num_books,
        dim=args.dim,
        item_bias_init=item_bias_init,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_users = torch.from_numpy(
        train_df["user_idx"].to_numpy(np.int64)
    ).long()

    train_items = torch.from_numpy(
        train_df["book_idx"].to_numpy(np.int64)
    ).long()

    train_dataset = TensorDataset(train_users, train_items)

    batch_size = min(args.batch_size, len(train_dataset))
    if batch_size <= 0:
        raise ValueError("No training data available.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    use_amp = (device.type == "cuda") and autocast is not None and GradScaler is not None
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"AMP enabled: {use_amp}")

    best_recall = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()

        loss_sum = 0.0
        steps = 0

        for users, items in train_loader:
            users = users.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    loss = in_batch_softmax_loss(
                        model=model,
                        users=users,
                        items=items,
                        item_log_probs=item_log_probs,
                        temperature=args.temperature,
                    )
            else:
                loss = in_batch_softmax_loss(
                    model=model,
                    users=users,
                    items=items,
                    item_log_probs=item_log_probs,
                    temperature=args.temperature,
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_sum += loss.item()
            steps += 1

        avg_loss = loss_sum / max(steps, 1)
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={avg_loss:.4f} "
            f"time={elapsed:.1f}s"
        )

        recall = masked_recall_at_k(
            model=model,
            val=val,
            seen_tensors=seen_tensors,
            temperature=args.temperature,
            k=args.k,
            eval_batch_size=args.eval_batch,
            device=device,
        )

        best_recall = max(best_recall, recall)

        print(f"Validation masked Recall@{args.k}: {recall:.4f}")

        if recall >= args.target_recall:
            print(f"Target reached: Recall@{args.k} >= {args.target_recall:.2f}")
            break

    print(f"Best validation masked Recall@{args.k}: {best_recall:.4f}")


if __name__ == "__main__":
    main()