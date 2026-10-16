import argparse
import gc
import math
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch

from model import Att_Diffuse_model, create_model_diffu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s_dataset", default="movie")
    parser.add_argument("--t_dataset", default="video")
    parser.add_argument("--data_root", default="../data")
    parser.add_argument(
        "--traditional_checkpoint",
        default="./saved_model/movie_video/traditional_target.pth",
    )
    parser.add_argument(
        "--debias_checkpoint",
        default="./saved_model/movie_video/debias_target.pth",
    )
    parser.add_argument("--output_csv", default="fusion_results.csv")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--random_seed", type=int, default=1997)
    parser.add_argument("--eval_seed", type=int, default=2026)
    parser.add_argument("--candidate_count", type=int, default=101)
    parser.add_argument("--popular_ratio", type=float, default=0.2)
    parser.add_argument("--metric_ks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.0, 0.1595, 0.25, 0.5, 0.75, 1.0],
        help="0=traditional only; 1=debias only",
    )

    # Model settings must match training.
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb_dropout", type=float, default=0.3)
    parser.add_argument("--hidden_act", default="gelu")
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--schedule_sampler_name", default="lossaware")
    parser.add_argument("--diffusion_steps", type=int, default=32)
    parser.add_argument("--lambda_uncertainty", type=float, default=0.001)
    parser.add_argument("--noise_schedule", default="trunc_lin")
    parser.add_argument("--rescale_timesteps", action="store_true", default=True)
    parser.add_argument("--source_item_num", type=int, default=98507)
    parser.add_argument("--target_item_num", type=int, default=23978)
    parser.add_argument("--num_gpu", type=int, default=1)
    return parser.parse_args()


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_popularity_groups(train_data, popular_ratio):
    counter = Counter()
    for seq in train_data["seq"]:
        counter.update(int(item) for item in seq if int(item) != 0)
    counter.update(int(item) for item in train_data["next"] if int(item) != 0)

    sorted_items = sorted(counter, key=lambda item: (-counter[item], item))
    num_popular = max(1, int(math.ceil(len(sorted_items) * popular_ratio)))
    observed = set(sorted_items)
    popular = set(sorted_items[:num_popular])
    unpopular_seen = observed - popular
    return popular, unpopular_seen, observed

def build_candidates(test_data, item_num, candidate_count, seed):
    """Column 0 is always the ground-truth item."""
    rng = random.Random(seed)
    all_candidates = []
    for target in test_data["next"]:
        target = int(target)
        candidates = [target]
        used = {target}
        while len(candidates) < candidate_count:
            item = rng.randint(0, item_num - 1)
            if item not in used:
                used.add(item)
                candidates.append(item)
        all_candidates.append(candidates)
    return torch.tensor(all_candidates, dtype=torch.long)

def strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict

def load_model(checkpoint, args):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    args.item_num = args.source_item_num
    model = Att_Diffuse_model(create_model_diffu(args), args)
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(args.device)
    model.eval()
    return model

def collect_candidate_probabilities(model, test_data, candidates, args):
    """Return [number of samples, candidate_count] normalized probabilities."""
    fix_seed(args.eval_seed)
    batches = []

    with torch.no_grad():
        for start in range(0, len(test_data), args.batch_size):
            end = min(start + args.batch_size, len(test_data))
            batch = test_data.iloc[start:end]
            seq = torch.as_tensor(batch["seq"].tolist(), dtype=torch.long, device=args.device)
            target = torch.as_tensor(
                batch["next"].tolist(), dtype=torch.long, device=args.device
            ).unsqueeze(1)
            candidate_ids = candidates[start:end].to(args.device)

            _, rep_diffu, _, _, _, _, _ = model(
                seq, target, None, False, args, 0.2, train_flag=False
            )
            all_scores = model.diffu_rep_pre(rep_diffu, False)
            candidate_scores = torch.gather(all_scores, 1, candidate_ids)

            # Normalize each model before fusion because raw logit scales may differ.
            batches.append(torch.softmax(candidate_scores, dim=1).cpu())

    return torch.cat(batches, dim=0)

def group_metrics(scores, candidates, masks, ks, popular_items, target_item_num):
    max_k = max(ks)
    top_positions = torch.topk(scores, k=max_k, dim=1).indices
    positive_position = torch.zeros(len(scores), dtype=torch.long)
    hit = top_positions.eq(positive_position.unsqueeze(1))
    top_items = torch.gather(candidates, 1, top_positions)

    rows = {}
    for group_name, mask in masks.items():
        count = int(mask.sum().item())
        result = {"Samples": count}
        for k in ks:
            if count == 0:
                result[f"HR@{k}"] = None
                result[f"NDCG@{k}"] = None
                continue
            group_hit = hit[mask, :k]
            hr = group_hit.any(dim=1).float().mean().item() * 100
            discounts = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float))
            ndcg = (group_hit.float() * discounts).sum(dim=1).mean().item() * 100
            result[f"HR@{k}"] = round(hr, 4)
            result[f"NDCG@{k}"] = round(ndcg, 4)
        rows[group_name] = result

    popular_lookup = torch.zeros(target_item_num + 1, dtype=torch.bool)
    valid_popular = [item for item in popular_items if 0 <= item < len(popular_lookup)]
    popular_lookup[torch.tensor(valid_popular, dtype=torch.long)] = True
    for k in ks:
        ratio = popular_lookup[top_items[:, :k]].float().mean().item() * 100
        rows["Overall"][f"PopularRatio@{k}"] = round(ratio, 4)
    return rows

def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")
    if any(beta < 0 or beta > 1 for beta in args.betas):
        raise ValueError("Every beta must be between 0 and 1.")
    if args.candidate_count <= max(args.metric_ks):
        raise ValueError("candidate_count must be larger than the largest K.")

    target_dir = os.path.join(args.data_root, args.t_dataset)
    train_data = pd.read_pickle(os.path.join(target_dir, "train.df"))
    test_data = pd.read_pickle(os.path.join(target_dir, "test.df")).reset_index(drop=True)

    popular, unpopular_seen, observed = build_popularity_groups(
        train_data, args.popular_ratio
    )
    candidates = build_candidates(
        test_data, args.target_item_num, args.candidate_count, args.random_seed
    )
    targets = torch.tensor(test_data["next"].astype(int).tolist(), dtype=torch.long)
    masks = {
        "Overall": torch.ones(len(test_data), dtype=torch.bool),
        "Popular": torch.tensor([item in popular for item in targets.tolist()]),
        "Unpopular-seen": torch.tensor(
            [item in unpopular_seen for item in targets.tolist()]
        ),
        "Unseen": torch.tensor([item not in observed for item in targets.tolist()]),
    }

    print("Loading traditional model...")
    traditional_model = load_model(args.traditional_checkpoint, args)
    traditional_prob = collect_candidate_probabilities(
        traditional_model, test_data, candidates, args
    )
    del traditional_model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    print("Loading debias model...")
    debias_model = load_model(args.debias_checkpoint, args)
    debias_prob = collect_candidate_probabilities(
        debias_model, test_data, candidates, args
    )
    del debias_model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    all_rows = []
    for beta in args.betas:
        fused = (1.0 - beta) * traditional_prob + beta * debias_prob
        metrics = group_metrics(
            fused,
            candidates,
            masks,
            args.metric_ks,
            popular,
            args.target_item_num,
        )
        for group, values in metrics.items():
            all_rows.append({"Beta": beta, "Group": group, **values})

    result = pd.DataFrame(all_rows)
    result.to_csv(args.output_csv, index=False)

    display_columns = [
        "Beta", "Group", "Samples",
        "HR@5", "HR@10", "HR@20",
        "NDCG@5", "NDCG@10", "NDCG@20",
        "PopularRatio@5", "PopularRatio@10", "PopularRatio@20"
    ]
    display_columns = [column for column in display_columns if column in result.columns]
    print("\nFusion result: beta=0 is traditional; beta=1 is debias")
    print(result[display_columns].to_string(index=False))
    print(f"\nFull results saved to: {args.output_csv}")


if __name__ == "__main__":
    args = parse_args()
    main(args)