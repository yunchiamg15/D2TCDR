"""Only method-specific augmentation decisions live here."""
import hashlib
import math
import random

import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from rng_utils import isolated_rng
from .high_hpr_b3 import augment_selected_b3
from .reward import RewardBaseline
from .high_hpr_state import SequenceState

LEARNED = {'v3_selection', 'copy_replace', 'four_head'}


def build_high_hpr_pool(train, popular, tail, ratio, seed):
    """Rank original 90% train Tail-next once; seeded ties; stable row IDs."""
    if not 0 < ratio <= 1:
        raise ValueError('Pool ratio must be in (0, 1]')
    rows = []
    rng = random.Random(seed)
    for _, row in train.iterrows():
        if int(row['next']) not in tail:
            continue
        history = [int(v) for v in row['seq'] if int(v) != 0]
        hpr = sum(v in popular for v in history) / len(history) if history else 0.
        rows.append((int(row['_row_id']), hpr, rng.random()))
    rows.sort(key=lambda x: (-x[1], x[2], x[0]))
    count = min(len(rows), math.floor(len(rows) * ratio))
    ids = {v[0] for v in rows[:count]}
    return ids, dict(tail_next=len(rows), pool_size=count, pool_ratio=ratio,
                    cutoff_hpr=rows[count-1][1] if count else None,
                    eligible_ids=sorted(ids), tie_seed=seed, rounding='floor')


def empty_stats():
    return dict(tail_samples=0, eligible_samples=0, decisions=0, copies=0, replaced=0, full_copies=0,
        skipped=0, no_legal=0, too_short=0, no_older_popular=0, no_candidate=0,
        selected_ids=[], positions={}, candidate_groups={'Popular': 0, 'Tail': 0},
        candidate_count_sum=0, candidate_pool_popular_count=0,
        copy_probability_mean=None, copy_rate=None)


class DebiasAgent(nn.Module):
    def __init__(self, method, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        if method == 'v3_selection':
            self.selection_head = nn.Linear(hidden, 1)
        elif method == 'copy_replace':
            self.copy_head = nn.Linear(hidden, 2)
        elif method == 'four_head':
            self.copy_head = nn.Linear(hidden, 2)
            self.replace_head = nn.Linear(hidden, 2)
            self.position_head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.candidate_head = nn.Sequential(nn.Linear(hidden + 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            raise ValueError('No learned policy for this method')


class AugmentationController(SequenceState):
    def __init__(self, train, popular, tail, candidates, args):
        super().__init__(train, popular, tail, candidates, args)
        self.method = args.method
        self.eligible_ids, self.pool_metadata = build_high_hpr_pool(
            train, self.popular, self.tail, args.high_hpr_pool_ratio, args.random_seed)
        self.policy_seed = args.target_seed + 8000
        self.carry = 0.
        self.policy = None
        if self.method in LEARNED:
            with isolated_rng(args.target_seed + 8000):
                self.policy = DebiasAgent(self.method, args.agent_hidden).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.agent_lr)
            self.baseline = RewardBaseline(args.reward_baseline_window)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(args.target_seed + 9000)
        self._log_probs = []
        self._entropies = []
        self._normalized_entropies = []

    def draw(self, logits):
        distribution = Categorical(logits=logits)
        action = torch.multinomial(distribution.probs, 1, generator=self.generator).squeeze(0)
        if self.policy is not None:
            self._log_probs.append(distribution.log_prob(action))
            self._entropies.append(distribution.entropy())
            if logits.numel() > 1:
                self._normalized_entropies.append(distribution.entropy() / math.log(logits.numel()))
        return int(action.item())

    def edit_rng(self, epoch, row_id):
        # Common random positions/items for the same original sample in the same
        # epoch; selecting other samples cannot advance this sample's RNG stream.
        token = f'{self.args.augmentation_seed}:{epoch}:{row_id}'.encode()
        return random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8], 'big'))

    def fixed_budget_selection(self, batch, tail_positions):
        n = len(tail_positions)
        budget = self.args.agent_budget_ratio * n + self.carry
        k = min(int(budget + 1e-9), n)
        self.carry = budget - k
        if n == 0 or k == 0:
            return []
        if self.policy is None:
            logits = torch.zeros(n, device=self.device)
        else:
            values = [self.sample_features(batch.iloc[i]['seq'], int(batch.iloc[i]['next']))
                      for i in tail_positions]
            states = torch.tensor(values, dtype=torch.float32, device=self.device)
            logits = self.policy.selection_head(self.policy.encoder(states)).squeeze(-1)
        selected = []
        for _ in range(k):
            local = self.draw(logits)
            selected.append(tail_positions[local])
            logits = logits.masked_fill(torch.arange(n, device=self.device) == local, float('-inf'))
        return selected

    def random_b3_copy(self, row, epoch, stats):
        duplicate = row.copy()
        duplicate['seq'] = list(row['seq'])
        duplicate['is_augmented'] = 1
        changed, detail = augment_selected_b3(pd.DataFrame([duplicate]), self.popular,
            self.candidates, self.edit_rng(epoch, int(row['_row_id'])),
            preserve_recent=self.args.preserve_recent, replacement_topk=self.args.replacement_topk)
        for key in ('too_short', 'no_older_popular', 'no_candidate'):
            stats[key] += detail[key]
        return changed.iloc[0].copy()

    def chosen_copy(self, row, t, chosen):
        duplicate = row.copy()
        duplicate['seq'] = list(row['seq'])
        duplicate['is_augmented'] = 1
        changed, _ = augment_selected_b3(pd.DataFrame([duplicate]), self.popular,
            self.candidates, None, preserve_recent=self.args.preserve_recent,
            replacement_topk=self.args.replacement_topk, decisions=[(t, chosen)])
        return changed.iloc[0].copy()

    def count_copy(self, row, duplicate, stats):
        stats['copies'] += 1
        stats['selected_ids'].append(int(row['_row_id']))
        before = [int(v) for v in row['seq'] if int(v) != 0]
        after = [int(v) for v in duplicate['seq'] if int(v) != 0]
        changed = [i for i, (u, v) in enumerate(zip(before, after)) if u != v]
        if not changed:
            stats['full_copies'] += 1
            return
        if len(changed) != 1 or len(before) != len(after):
            raise ValueError('Augmentation must replace exactly one valid item')
        t = changed[0]
        stats['replaced'] += 1
        stats['positions'][str(t + 1)] = stats['positions'].get(str(t + 1), 0) + 1
        group = 'Popular' if after[t] in self.popular else 'Tail'
        stats['candidate_groups'][group] += 1
        blocked = set(before) | {0, int(row['next'])}
        items = [v for v, _ in self.candidates.get(before[t], []) if v not in blocked][:self.args.replacement_topk]
        stats['candidate_count_sum'] += len(items)
        stats['candidate_pool_popular_count'] += sum(v in self.popular for v in items)

    def augment(self, batch, epoch):
        if batch['is_augmented'].astype(int).ne(0).any():
            raise ValueError('Start every decision from original samples')
        self._log_probs, self._entropies, self._normalized_entropies = [], [], []
        stats = empty_stats()
        all_tail_positions = [i for i, y in enumerate(batch['next']) if int(y) in self.tail]
        tail_positions = [i for i in all_tail_positions
                          if int(batch.iloc[i]['_row_id']) in self.eligible_ids]
        stats['tail_samples'] = len(all_tail_positions)
        stats['eligible_samples'] = len(tail_positions)
        copies = []
        probabilities = []
        if self.method in ('model_b', 'b3_full'):
            for i in tail_positions:
                row = batch.iloc[i]
                if self.method == 'model_b':
                    duplicate = row.copy()
                    duplicate['seq'] = list(row['seq'])
                    duplicate['is_augmented'] = 1
                else:
                    duplicate = self.random_b3_copy(row, epoch, stats)
                self.count_copy(row, duplicate, stats)
                copies.append(duplicate)
        elif self.method in ('random_b3', 'v3_selection'):
            selected = self.fixed_budget_selection(batch, tail_positions)
            stats['decisions'] = len(selected)  # quota draws, not binary sample decisions
            for i in selected:
                row = batch.iloc[i]
                duplicate = self.random_b3_copy(row, epoch, stats)
                self.count_copy(row, duplicate, stats)
                copies.append(duplicate)
        elif self.method in ('copy_replace', 'four_head'):
            for i in tail_positions:
                row = batch.iloc[i]
                history, features, options = self.options_for(row['seq'], int(row['next']))
                if not options:
                    stats['no_legal'] += 1
                    if self.method == 'copy_replace':
                        continue  # action 1 is impossible; no unchanged copies in this method
                encoded = self.policy.encoder(torch.tensor(features, dtype=torch.float32, device=self.device))
                logits = self.policy.copy_head(encoded)
                probabilities.append(float(logits.softmax(-1)[1].detach()))
                stats['decisions'] += 1
                if self.draw(logits) == 0:
                    continue
                if self.method == 'copy_replace':
                    rng = self.edit_rng(epoch, int(row['_row_id']))
                    t, _, items, _ = rng.choice(options)
                    duplicate = self.chosen_copy(row, t, rng.choice(items))
                elif not options:
                    duplicate = row.copy()
                    duplicate['seq'] = list(row['seq'])
                    duplicate['is_augmented'] = 1
                elif self.draw(self.policy.replace_head(encoded)) == 0:
                    duplicate = row.copy()
                    duplicate['seq'] = list(row['seq'])
                    duplicate['is_augmented'] = 1
                else:
                    pos_values = torch.tensor([[opt[1]] for opt in options], dtype=torch.float32, device=self.device)
                    pos_logits = self.policy.position_head(torch.cat([encoded.expand(len(options), -1), pos_values], -1)).squeeze(-1)
                    option = options[self.draw(pos_logits)]
                    t, _, items, values = option
                    candidate_values = torch.tensor(values, dtype=torch.float32, device=self.device)
                    candidate_logits = self.policy.candidate_head(torch.cat([encoded.expand(len(items), -1), candidate_values], -1)).squeeze(-1)
                    duplicate = self.chosen_copy(row, t, items[self.draw(candidate_logits)])
                self.count_copy(row, duplicate, stats)
                copies.append(duplicate)
        else:
            raise ValueError(f'Unknown augmentation method: {self.method}')
        stats['skipped'] = len(tail_positions) - stats['copies']
        stats['budget_carry'] = self.carry
        stats['copy_probability_mean'] = sum(probabilities) / len(probabilities) if probabilities else None
        stats['copy_rate'] = stats['copies'] / stats['tail_samples'] if stats['tail_samples'] else None
        augmented = (pd.concat([batch, pd.DataFrame(copies, columns=batch.columns)], ignore_index=True)
                     if copies else batch.copy().reset_index(drop=True))
        log_prob = torch.stack(self._log_probs).sum() if self._log_probs else None
        entropy = torch.stack(self._entropies).mean() if self._entropies else None
        normalized = (float(torch.stack(self._normalized_entropies).mean().detach())
                      if self._normalized_entropies else None)
        stats['normalized_entropy'] = normalized
        return augmented, log_prob, entropy, stats

    def update(self, reward, log_prob, entropy):
        if log_prob is None:
            return None
        advantage = reward - self.baseline.previous_average(reward)
        loss = -advantage * log_prob - self.args.agent_entropy_coef * entropy
        if not torch.isfinite(loss):
            raise ValueError('Non-finite policy loss')
        self.optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.)
        grads = {name: sum(float(p.grad.detach().norm()) for p in module.parameters() if p.grad is not None)
                 for name, module in self.policy.named_children()}
        self.optimizer.step()
        self.baseline.update(reward)
        return dict(reward=float(reward), advantage=float(advantage), policy_loss=float(loss.detach()),
                    entropy=float(entropy.detach()), grad_norm=float(norm), head_grad_norms=grads)
