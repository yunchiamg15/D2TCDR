
from collections import deque
import torch


class RewardBaseline:

    def __init__(self, window=50):
        self.values = deque(maxlen=window)

    def previous_average(self, reward):
        if not self.values:
            return reward

        return (
            sum(self.values) / len(self.values)
        )

    def update(self, reward):
        self.values.append(float(reward))


def calculate_reward(
    metrics,
    ref_overall,
    ref_popular,
    lambda_overall=1.0,
    lambda_popular=1.0,
    epsilon_overall=0.002,
    epsilon_popular=0.002
):

    tail_hr = metrics["Tail_HR@10"]
    overall_hr = metrics["Overall_HR@10"]
    popular_hr = metrics["Popular_HR@10"]

    overall_penalty = max(
        0.0,
        ref_overall - overall_hr - epsilon_overall
    )

    popular_penalty = max(
        0.0,
        ref_popular - popular_hr - epsilon_popular
    )

    reward = (
        tail_hr
        - lambda_overall * overall_penalty
        - lambda_popular * popular_penalty
    )

    return reward


@torch.no_grad()
def evaluate_reward(
    model,
    reward_data,
    args,
    popular_items,
    tail_items,
    epoch=0,
    random_seed=42
):

    device = torch.device(args.device)

    was_training = model.training
    model.eval()

    counts = {
        "Overall": 0,
        "Popular": 0,
        "Tail": 0
    }

    hits = {
        "Overall": 0.0,
        "Popular": 0.0,
        "Tail": 0.0
    }

    cuda_devices = (
        [torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )

    try:
        # Use fixed diffusion evaluation randomness
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(random_seed)

            for start in range(
                0,
                len(reward_data),
                args.batch_size
            ):

                batch_df = reward_data.iloc[
                    start:start + args.batch_size
                ]

                seq = torch.as_tensor(
                    batch_df["seq"].tolist(),
                    dtype=torch.long,
                    device=device
                )

                targets = torch.as_tensor(
                    batch_df["next"].tolist(),
                    dtype=torch.long,
                    device=device
                ).view(-1, 1)

                candidates = torch.as_tensor(
                    batch_df["negative_samples"].tolist(),
                    dtype=torch.long,
                    device=device
                )

                # Positive candidate is at column 0
                if not torch.all(
                    candidates[:, 0] == targets.view(-1)
                ):
                    raise ValueError(
                        "Reward positive candidate mismatch."
                    )

                _, diffu_rep, _, _, _, _, _ = model(
                    seq,
                    targets,
                    None,
                    False,
                    args,
                    epoch,
                    train_flag=False
                )

                scores = model.diffu_rep_pre(
                    diffu_rep,
                    pretrain_flag=False
                )

                # Scores of 1 positive + 100 negatives
                candidate_scores = scores.gather(
                    1, candidates
                )

                topk_indices = (
                    candidate_scores
                    .topk(k=10, dim=1)
                    .indices
                )

                batch_hits = (
                    (topk_indices == 0)
                    .any(dim=1)
                    .float()
                )

                target_list = (
                    targets.view(-1).tolist()
                )

                popular_mask = torch.tensor(
                    [
                        item in popular_items
                        for item in target_list
                    ],
                    dtype=torch.bool,
                    device=device
                )

                tail_mask = torch.tensor(
                    [
                        item in tail_items
                        for item in target_list
                    ],
                    dtype=torch.bool,
                    device=device
                )

                counts["Overall"] += len(batch_df)
                hits["Overall"] += (
                    batch_hits.sum().item()
                )

                counts["Popular"] += (
                    popular_mask.sum().item()
                )

                hits["Popular"] += (
                    batch_hits[popular_mask].sum().item()
                )

                counts["Tail"] += (
                    tail_mask.sum().item()
                )

                hits["Tail"] += (
                    batch_hits[tail_mask].sum().item()
                )

    finally:
        model.train(was_training)

    if (
        counts["Popular"] == 0
        or counts["Tail"] == 0
    ):
        raise ValueError(
            "Reward Set must contain both "
            "Popular and Tail samples."
        )

    return {
        "Overall_HR@10": (
            hits["Overall"] / counts["Overall"]
        ),
        "Popular_HR@10": (
            hits["Popular"] / counts["Popular"]
        ),
        "Tail_HR@10": (
            hits["Tail"] / counts["Tail"]
        ),
        "counts": counts
    }
