
import torch
import torch.nn as nn

from torch.distributions import Categorical


class DebiasAgent(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Shared Policy Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action B: Oversampling Selection
        self.selection_head = nn.Linear(
            hidden_dim, 1
        )

    def forward(self, states):

        features = self.encoder(states)

        selection_logits = (
            self.selection_head(features)
            .squeeze(-1)
        )

        return selection_logits


def sample_actions(
    agent,
    states,
    eligible_mask,
    budget_carry=0.0,
    budget_ratio=0.20
):
    """
    Returns:
        duplicate_idx:
            original sample indices to duplicate

        joint_log_prob:
            log probability of the selection action

        next_carry:
            fractional oversampling budget
    """

    device = states.device

    tail_indices = torch.where(
        eligible_mask
    )[0]

    n_tail = tail_indices.numel()

    if n_tail == 0:
        return tail_indices, None, budget_carry

    tail_states = states[tail_indices]

    selection_logits = agent(
        tail_states
    )

    # ==============================
    # Action B: Oversampling
    # ==============================

    budget = (
        budget_ratio * n_tail
        + budget_carry
    )

    k = min(int(budget + 1e-9), n_tail)

    next_carry = budget - k

    remaining_logits = selection_logits

    selected_indices = []

    log_prob_b = (
        selection_logits.sum() * 0.0
    )

    # Stochastic sampling without replacement
    for _ in range(k):

        distribution = Categorical(
            logits=remaining_logits
        )

        chosen = distribution.sample()

        selected_indices.append(chosen)

        log_prob_b = (
            log_prob_b
            + distribution.log_prob(chosen)
        )

        selection_mask = (
            torch.arange(
                n_tail,
                device=device
            ) == chosen
        )

        remaining_logits = (
            remaining_logits.masked_fill(
                selection_mask,
                float("-inf")
            )
        )

    if selected_indices:

        selected_local = torch.stack(
            selected_indices
        )

        duplicate_idx = tail_indices[
            selected_local
        ]

    else:
        duplicate_idx = tail_indices[:0]

    return duplicate_idx, log_prob_b, next_carry
