import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(path, epoch):
    counts = Counter()
    with path.open(encoding='utf-8') as log:
        metadata = json.loads(next(log))
        for line in log:
            batch = json.loads(line)
            if epoch is not None and batch['epoch'] != epoch:
                continue
            stats = batch['augmentation']
            events = stats.get('replacements')
            if events is None:
                raise ValueError('This log predates per-replacement logging. Run the updated code to collect it.')
            counts['batches'] += 1
            for field in ('eligible_samples', 'copies', 'replaced', 'full_copies'):
                counts[field] += stats[field]
            for event in events:
                counts[event['replacement_group']] += 1
                counts['outside_b3'] += not event['in_b3_topk']
                counts['no_b3_candidates'] += event['b3_candidate_count'] == 0

    if counts['batches'] == 0:
        raise ValueError('No batches match the requested epoch.')
    replacements = counts['replaced']
    return {
        'log': str(path),
        'method': metadata['method'],
        'epoch': epoch if epoch is not None else 'all',
        'batches': counts['batches'],
        'eligible_samples': counts['eligible_samples'],
        'copies': counts['copies'],
        'replaced': replacements,
        'full_copies': counts['full_copies'],
        'replacement_popular': counts['Popular'],
        'replacement_tail': counts['Tail'],
        'outside_b3_topk': counts['outside_b3'],
        'outside_b3_topk_pct': round(100 * counts['outside_b3'] / replacements, 4) if replacements else None,
        'no_b3_candidates_at_position': counts['no_b3_candidates'],
        'b3_reference_topk': metadata['b3_reference_topk'],
        'b3_reference_min_cooccurrence': metadata['b3_reference_min_cooccurrence'],
    }


def main():
    parser = argparse.ArgumentParser(description='Summarize replacements in D2TCDR augmentation logs.')
    parser.add_argument('logs', nargs='+', type=Path)
    parser.add_argument('--epoch', type=int, help='Only include this zero-based training epoch.')
    args = parser.parse_args()
    for path in args.logs:
        try:
            print(json.dumps(summarize(path, args.epoch), ensure_ascii=False))
        except (OSError, StopIteration, KeyError, ValueError) as error:
            parser.exit(1, f'{path}: {error}\n')


if __name__ == '__main__':
    main()
