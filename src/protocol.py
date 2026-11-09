"""One immutable data/evaluation protocol shared by all five methods."""
import contextlib
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

VERSION = 'd2tcdr_shared_v1'
COMMON_KEYS = (
    'random_seed', 'target_seed', 'eval_seed', 'augmentation_seed', 'reward_holdout_ratio', 'popular_ratio',
    'source_item_num', 'target_item_num', 'max_len', 'batch_size', 'hidden_size',
    'hidden_act', 'num_blocks', 'dropout', 'emb_dropout', 'p', 'w', 'diffusion_steps',
    'schedule_sampler_name', 'noise_schedule', 'rescale_timesteps', 'lambda_uncertainty',
    'optimizer', 'lr', 'weight_decay', 'loss_lambda', 'decay_step', 'gamma', 'epochs', 'eval_interval', 'patience',
    'metric_ks', 'preserve_recent', 'replacement_topk', 'min_cooccurrence',
)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize_frame(frame, args):
    frame = frame.copy().reset_index(drop=True)
    if not {'seq', 'next'}.issubset(frame):
        raise ValueError('Data must contain seq and next')
    frame['seq'] = frame['seq'].map(lambda seq: [int(v) for v in seq])
    frame['next'] = frame['next'].astype(int)
    lengths = frame['seq'].map(lambda seq: sum(v != 0 for v in seq))
    for seq, target in zip(frame['seq'], frame['next']):
        if len(seq) != args.max_len or not any(seq):
            raise ValueError('Every padded history must have max_len entries and at least one valid item')
        if not 1 <= target <= args.target_item_num or any(v < 0 or v > args.target_item_num for v in seq):
            raise ValueError('Item IDs must be in 1..target_item_num; history padding is 0')
    # The model reads seq/next; derive effective lengths to avoid stale auxiliary columns.
    frame['len_seq'] = lengths.astype(int)
    return frame[['seq', 'next', 'len_seq']]


def frame_records(frame):
    return [[list(seq), int(target), int(length)]
            for seq, target, length in zip(frame['seq'], frame['next'], frame['len_seq'])]


def popularity_groups(train, ratio):
    count = Counter(v for seq in train['seq'] for v in seq if v != 0)
    count.update(int(v) for v in train['next'] if int(v) != 0)
    ranked = sorted(count, key=lambda v: (-count[v], v))
    popular = set(ranked[:max(1, math.ceil(len(ranked) * ratio))])
    tail = set(ranked) - popular
    if not tail:
        raise ValueError('Training data must have observed Tail items')
    return popular, tail


def candidate_rows(frame, item_num, seed):
    if item_num < 101:
        raise ValueError('Evaluation requires at least 101 non-padding items')
    rng = random.Random(seed)
    rows = []
    for target in frame['next']:
        values = [int(target)]
        seen = {int(target)}
        while len(values) < 101:
            value = rng.randint(1, item_num)
            if value not in seen:
                seen.add(value)
                values.append(value)
        rows.append(values)
    return rows


def prepare_protocol(train, val, test, args, source_sha):
    train, val, test = [normalize_frame(frame, args) for frame in (train, val, test)]
    config = {key: getattr(args, key) for key in COMMON_KEYS}
    inputs = dict(train=digest_json(frame_records(train)), val=digest_json(frame_records(val)),
                  test=digest_json(frame_records(test)), source_sha256=source_sha)
    code_root = Path(__file__).resolve().parent
    # Recommendation code changes require a new protocol, too.
    inputs['recommendation_code'] = {name: digest_file(code_root / name)
        for name in ('model.py', 'diffurec.py', 'step_sample.py') if (code_root / name).exists()}
    inputs['shared_code'] = {name: digest_file(code_root / name)
        for name in ('main.py', 'trainer.py', 'protocol.py', 'agent/b3.py', 'agent/reward.py')
        if (code_root / name).exists()}
    path = Path(args.protocol_file)
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        fingerprint = record.get('fingerprint')
        payload = {key: value for key, value in record.items() if key != 'fingerprint'}
        if fingerprint != digest_json(payload):
            raise ValueError('Shared protocol integrity check failed; do not edit the cached file')
        if record.get('version') != VERSION or record.get('config') != config or record.get('inputs') != inputs:
            raise ValueError('Shared protocol mismatch: keep data, Source and common settings identical; use a new output directory for a different experiment')
    else:
        res_indices = train.sample(frac=args.reward_holdout_ratio, random_state=args.random_seed).index.tolist()
        res_set = set(res_indices)
        main_indices = [i for i in range(len(train)) if i not in res_set]
        main = train.iloc[main_indices]
        res = train.iloc[res_indices]
        if len(main) < args.batch_size or res.empty or val.empty or test.empty:
            raise ValueError('Need at least one full training batch and nonempty RES/Validation/Test')
        popular, tail = popularity_groups(main, args.popular_ratio)
        targets = set(int(v) for v in res['next'])
        if not targets & popular or not targets & tail:
            raise ValueError('Fixed RES must contain both Popular and Tail targets')
        record = dict(version=VERSION, config=config, inputs=inputs,
            main_indices=main_indices, res_indices=res_indices,
            popular=sorted(popular), tail=sorted(tail),
            candidates=dict(val=candidate_rows(val, args.target_item_num, args.eval_seed + 11),
                            res=candidate_rows(res, args.target_item_num, args.eval_seed + 22),
                            test=candidate_rows(test, args.target_item_num, args.eval_seed + 33)))
        record['fingerprint'] = digest_json(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally overwriting another run's protocol.
        with path.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, separators=(',', ':'))
    main = train.iloc[record['main_indices']].copy().reset_index(drop=True)
    main['_row_id'] = record['main_indices']
    main['is_augmented'] = 0
    res = train.iloc[record['res_indices']].copy().reset_index(drop=True)
    for name, frame in [('val', val), ('res', res), ('test', test)]:
        frame['negative_samples'] = record['candidates'][name]
    return main, val, res, test, set(record['popular']), set(record['tail']), record


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def isolated_rng(seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            seed_all(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
