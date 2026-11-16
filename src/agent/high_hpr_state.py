"""Training-only features; High-HPR eligibility is fixed by the controller."""
import math
from collections import Counter
import torch

from .high_hpr_b3 import legal_b3_candidates


def build_target_frequency(train):
    frequency = Counter(int(v) for v in train['next'] if int(v) != 0)
    return frequency, max(frequency.values(), default=1)


class SequenceState:
    def __init__(self, train, popular, tail, candidates, args):
        self.args = args
        self.device = torch.device(args.device)
        self.popular, self.tail = set(popular), set(tail)
        self.candidates = candidates
        self.similarities = {u: dict(pairs) for u, pairs in candidates.items()}
        self.frequency, maximum = build_target_frequency(train)
        self.denominator = math.log1p(maximum)
        self.cache = {}

    def frequency_feature(self, item):
        return math.log1p(self.frequency.get(int(item), 0)) / self.denominator

    def similarity(self, left, right):
        return self.similarities.get(int(left), {}).get(int(right), 0.)

    def sample_features(self, seq, target):
        history = [int(v) for v in seq if int(v) != 0]
        hpr = sum(v in self.popular for v in history) / len(history) if history else 0.
        return [hpr, self.frequency_feature(target)]

    def older_popular_positions(self, history):
        length = len(history)
        return [(t, t / (length - 1))
                for t in range(length - self.args.preserve_recent)
                if history[t] in self.popular]

    def options_for(self, seq, target):
        key = (tuple(int(v) for v in seq), int(target))
        if key in self.cache:
            return self.cache[key]
        history = [v for v in key[0] if v != 0]
        features = self.sample_features(history, target)
        blocked = set(history) | {0, int(target)}
        recent = history[-2:]
        options = []
        for t, pos in self.older_popular_positions(history):
            old = history[t]
            items = legal_b3_candidates(self.candidates, old, blocked, self.args.replacement_topk)
            if not items:
                continue
            values = [[pos, self.similarity(old, v),
                       sum(self.similarity(v, r) for r in recent) / len(recent),
                       self.frequency_feature(v)] for v in items]
            options.append((t, pos, items, values))
        result = history, features, options
        self.cache[key] = result
        return result
