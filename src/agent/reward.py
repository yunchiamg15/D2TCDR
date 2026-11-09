"""Same Reward and prior moving-average baseline for every learned policy."""
from collections import deque
import math


class RewardBaseline:
    def __init__(self, window=50):
        self.values = deque(maxlen=window)
    def previous_average(self, reward):
        return sum(self.values) / len(self.values) if self.values else reward
    def update(self, reward):
        self.values.append(float(reward))


def calculate_reward(metrics, references, args):
    reward = (metrics['Tail_HR@10']
        - args.reward_lambda_overall * max(0., references['Overall_HR@10']
            - metrics['Overall_HR@10'] - args.reward_epsilon_overall)
        - args.reward_lambda_popular * max(0., references['Popular_HR@10']
            - metrics['Popular_HR@10'] - args.reward_epsilon_popular))
    if not math.isfinite(reward):
        raise ValueError('Non-finite Reward')
    return reward
