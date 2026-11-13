
from collections import Counter
import math
import torch


def build_target_frequency(train_data):
    """
    Count target frequency from training next items only.
    """

    counter = Counter(
        int(item)
        for item in train_data["next"]
        if int(item) != 0
    )

    if not counter:
        raise ValueError("Training targets are empty.")

    max_frequency = max(counter.values())

    return counter, max_frequency


def build_sample_states(
    batch_df,
    popular_items,
    tail_items,
    target_counter,
    max_frequency,
    device
):
    """
    State:
        [Historical Popular Ratio,
         Normalized Target Frequency]

    Eligible:
        target item belongs to Tail.
    """

    hpr_values = []
    frequency_values = []
    eligible_values = []

    denominator = math.log1p(max_frequency)

    for history, target in zip(
        batch_df["seq"],
        batch_df["next"]
    ):
        target = int(target)

        valid_history = [
            int(item)
            for item in history
            if int(item) != 0
        ]

        # State 1: HPR
        if valid_history:
            hpr = sum(
                item in popular_items
                for item in valid_history
            ) / len(valid_history)
        else:
            hpr = 0.0

        # State 2: Normalized Target Frequency
        target_frequency = target_counter.get(
            target, 0
        )

        normalized_frequency = (
            math.log1p(target_frequency)
            / denominator
        )

        # Tail-next Eligibility
        eligible = target in tail_items

        hpr_values.append(hpr)
        frequency_values.append(normalized_frequency)
        eligible_values.append(eligible)

    states = torch.tensor(
        list(zip(hpr_values, frequency_values)),
        dtype=torch.float32,
        device=device
    )

    eligible_mask = torch.tensor(
        eligible_values,
        dtype=torch.bool,
        device=device
    )

    return states, eligible_mask
