"""Training-only features shared by all policies; no HPR eligibility threshold."""
import math
from collections import Counter
import torch


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

    def options_for(self, seq, target):
        key = (tuple(int(v) for v in seq), int(target))
        if key in self.cache:
            return self.cache[key]
        history = [v for v in key[0] if v != 0]
        length = len(history)
        hpr = sum(v in self.popular for v in history) / length if length else 0.
        features = [hpr, self.frequency_feature(target)]
        blocked = set(history) | {0, int(target)}
        recent = history[-2:]
        options = []
        for t in range(max(0, length - self.args.preserve_recent)):
            old = history[t]
            if old not in self.popular:
                continue
            legal = [(int(v), float(sim)) for v, sim in self.candidates.get(old, [])
                     if int(v) not in blocked][:self.args.replacement_topk]
            if not legal:
                continue
            pos = t / (length - 1)
            values = [[pos, sim, sum(self.similarity(v, r) for r in recent) / len(recent),
                       self.frequency_feature(v)] for v, sim in legal]
            options.append((t, pos, [v for v, _ in legal], values))
        result = history, features, options
        self.cache[key] = result
        return result
