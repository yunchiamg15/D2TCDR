import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import random
import pandas as pd
from collections import Counter

def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError

def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr

def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg

def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel

def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  

def build_item_popularity_groups(
    train_data,
    popular_ratio=0.2
):
    """
    使用 target-domain training set 建立三種 item 集合。
    Popular：
        training data 中出現次數最高的前 popular_ratio。
    Unpopular-seen：
        training data 中出現過，但不屬於 Popular。
    Unseen：
        不在 observed_train_items 中。
        Unseen 集合本身會在測試時依 target 判斷，
        因為無法只靠 training data 預先列出所有測試 unseen item。
    Popularity 統計包含：
        1. training seq 中的 item
        2. training next 中的 item
    item 0 是 padding，不納入統計。
    """
    if not 0 < popular_ratio < 1:
        raise ValueError(
            "popular_ratio must be between 0 and 1, "
            f"got {popular_ratio}"
        )
    item_counter = Counter()
    # 1. 統計 training sequence 中出現的 item
    for seq in train_data['seq']:
        item_counter.update(
            int(item)
            for item in seq
            if int(item) != 0
        )
    # 2. 統計 training target 中出現的 item
    item_counter.update(
        int(item)
        for item in train_data['next']
        if int(item) != 0
    )
    if len(item_counter) == 0:
        raise ValueError(
            "No valid items found in training data."
        )
    # 出現次數由高至低；
    # 次數相同時用 item ID 排序，確保可重現
    sorted_items = sorted(
        item_counter.keys(),
        key=lambda item: (
            -item_counter[item],
            item
        )
    )
    num_popular = max(
        1,
        int(
            np.ceil(
                len(sorted_items) * popular_ratio
            )
        )
    )
    # training data 中出現過的全部 item
    observed_train_items = set(sorted_items)
    # observed items 中最熱門的前 20%
    popular_items = set(
        sorted_items[:num_popular]
    )
    # training 中出現過，但不屬於 popular
    unpopular_seen_items = (
        observed_train_items - popular_items
    )
    return (
        popular_items,
        unpopular_seen_items,
        observed_train_items,
        item_counter
    )

def pad_or_truncate_sequence(seq, max_len):
    """
    移除 padding 0，保留最近 max_len 個 item，
    最後從左側重新補 0。
    """
    real_items = [
        int(item)
        for item in seq
        if int(item) != 0
    ]
    real_items = real_items[-max_len:]
    return (
        [0] * (max_len - len(real_items))
        + real_items
    )

def augment_tail_sequence(
    seq,
    max_len,
    drop_probability,
    preserve_recent,
    rng
):
    """
    對 Unpopular-seen target 的歷史 sequence
    做 context dropout。
    規則：
    1. 只刪除較早的歷史 item
    2. 最近 preserve_recent 個 item 固定保留
    3. 不打亂 item 順序
    4. 不修改 ground-truth next
    """
    if not 0 <= drop_probability <= 1:
        raise ValueError(
            'tail_drop_probability must be between 0 and 1.'
        )
    if preserve_recent < 1:
        raise ValueError(
            'tail_preserve_recent must be at least 1.'
        )
    real_items = [
        int(item)
        for item in seq
        if int(item) != 0
    ]
    # 序列太短，沒有較早 item 可以刪除
    if len(real_items) <= preserve_recent:
        return pad_or_truncate_sequence(
            real_items,
            max_len
        )
    older_items = real_items[:-preserve_recent]
    recent_items = real_items[-preserve_recent:]
    # 只對較舊的 item 做 dropout
    kept_older_items = []
    dropped_any_item = False
    for item in older_items:
        if rng.random() < drop_probability:
            dropped_any_item = True
        else:
            kept_older_items.append(item)
    # 若這次剛好一個都沒刪，強制隨機刪除一個較舊 item
    if not dropped_any_item and len(older_items) > 0:
        drop_index = rng.randrange(
            len(older_items)
        )
        kept_older_items = (
            older_items[:drop_index]
            + older_items[drop_index + 1:]
        )
    augmented_items = (
        kept_older_items
        + recent_items
    )
    return pad_or_truncate_sequence(
        augmented_items,
        max_len
    )

def build_doubled_unpopular_seen_train_data(
    train_data,
    unpopular_seen_items,
    max_len,
    drop_probability,
    preserve_recent,
    random_seed
):
    """
    保留所有原始 training rows。
    對每一筆 next 屬於 Unpopular-seen 的資料，
    額外產生一筆 context-dropout copy。
    Popular：
        只保留原始資料。
    Unpopular-seen：
        一筆原始資料 + 一筆 augmented 資料。
    """
    original_data = (
        train_data
        .copy()
        .reset_index(drop=True)
    )
    # 用來辨認原始資料與 augmented copy
    original_data['is_augmented'] = 0
    rng = random.Random(random_seed)
    augmented_rows = []
    unchanged_augmented_count = 0
    for _, row in original_data.iterrows():
        target_item = int(row['next'])
        # 只對 Unpopular-seen target 新增 copy
        if target_item not in unpopular_seen_items:
            continue
        augmented_row = row.copy()
        original_seq = list(row['seq'])
        normalized_original_seq = (
            pad_or_truncate_sequence(
                original_seq,
                max_len
            )
        )
        augmented_seq = augment_tail_sequence(
            seq=original_seq,
            max_len=max_len,
            drop_probability=drop_probability,
            preserve_recent=preserve_recent,
            rng=rng
        )
        if augmented_seq == normalized_original_seq:
            unchanged_augmented_count += 1
        # 只修改 sequence
        augmented_row['seq'] = augmented_seq
        # ground-truth next 維持原樣
        augmented_row['next'] = target_item
        if 'len_seq' in original_data.columns:
            augmented_row['len_seq'] = sum(
                int(item) != 0
                for item in augmented_seq
            )
        augmented_row['is_augmented'] = 1
        augmented_rows.append(
            augmented_row
        )
    augmented_data = pd.DataFrame(
        augmented_rows,
        columns=original_data.columns
    )
    doubled_train_data = pd.concat(
        [
            original_data,
            augmented_data
        ],
        ignore_index=True
    )
    # 打亂原始與 augmented rows
    doubled_train_data = (
        doubled_train_data
        .sample(
            frac=1,
            random_state=random_seed
        )
        .reset_index(drop=True)
    )
    original_tail_count = (
        original_data['next']
        .astype(int)
        .isin(unpopular_seen_items)
        .sum()
    )
    final_tail_count = (
        doubled_train_data['next']
        .astype(int)
        .isin(unpopular_seen_items)
        .sum()
    )
    stats = {
        'original_training_samples': (
            len(original_data)
        ),
        'original_tail_samples': int(
            original_tail_count
        ),
        'augmented_tail_copies': (
            len(augmented_data)
        ),
        'final_training_samples': (
            len(doubled_train_data)
        ),
        'final_tail_samples': int(
            final_tail_count
        ),
        'original_tail_ratio': round(
            original_tail_count
            / len(original_data),
            4
        ),
        'final_tail_ratio': round(
            final_tail_count
            / len(doubled_train_data),
            4
        ),
        'unchanged_augmented_sequences': (
            unchanged_augmented_count
        )
    }
    return doubled_train_data, stats

def per_sample_metrics(scores, labels, ks):
    """
    計算每一筆 test sample 個別的 HR@K 與 NDCG@K。
    scores:
        shape = [batch_size, item_num]
    labels:
        shape = [batch_size, 1]
    """
    labels = labels.view(-1)
    # 找出每筆 sample 分數最高的 Top-K items
    topk_indices = torch.topk(
        scores,
        k=max(ks),
        dim=-1
    ).indices
    # 判斷 Top-K 中是否包含真實 next item
    hit = topk_indices.eq(
        labels.unsqueeze(1)
    )
    metrics = {}
    for k in ks:
        hit_k = hit[:, :k]
        # 如果真實 item 出現在 Top-K，HR = 1
        # 否則 HR = 0
        metrics[f'HR@{k}'] = (
            hit_k.any(dim=1)
            .float()
            .detach()
            .cpu()
        )
        # rank 1、2、3...的折扣
        discounts = 1.0 / torch.log2(
            torch.arange(
                2,
                k + 2,
                device=scores.device,
                dtype=torch.float32
            )
        )
        # 真實 item 排得越前面，NDCG 越高
        metrics[f'NDCG@{k}'] = (
            hit_k.float()
            * discounts.unsqueeze(0)
        ).sum(dim=1).detach().cpu()
    return metrics, topk_indices

def init_metric_sums(ks):
    """
    建立用來累加 HR/NDCG 的字典。
    """
    metric_sums = {}
    for k in ks:
        metric_sums[f'HR@{k}'] = 0.0
        metric_sums[f'NDCG@{k}'] = 0.0
    return metric_sums

def finalize_metric_sums(
    metric_sums,
    sample_count
):
    """
    將指標加總除以 sample 數量，
    再轉成百分比。
    """
    if sample_count == 0:
        return {
            key: None
            for key in metric_sums
        }
    return {
        key: round(
            value / sample_count * 100,
            4
        )
        for key, value in metric_sums.items()
    }

def perturb_sequence_batch(
    sequences,
    targets,
    unpopular_seen_items,
    args,
    rng,
):
    """
    B1/B2/B3：動態修改當次訓練輸入。
    不新增資料，不修改 next，不改變 item 順序。
    """
    variant = args.sequence_variant
    probability = args.sequence_aug_prob
    preserve_recent = args.sequence_preserve_recent
    output_sequences = []
    eligible_count = 0
    changed_count = 0
    for seq, target in zip(sequences, targets):
        original_seq = list(seq)
        real_items = [
            int(item)
            for item in original_seq
            if int(item) != 0
        ]
        if variant == 'b3':
            eligible = True
        elif variant in ('b1', 'b2'):
            eligible = int(target) in unpopular_seen_items
        else:
            eligible = False
        # 只有歷史夠長才允許修改
        if not eligible or len(real_items) <= preserve_recent:
            output_sequences.append(original_seq)
            continue
        eligible_count += 1
        # 未抽中：本次使用原始 sequence
        if rng.random() >= probability:
            output_sequences.append(original_seq)
            continue
        if variant in ('b1', 'b3'):
            # 最近 preserve_recent 個 item 不可刪除
            older_count = len(real_items) - preserve_recent
            remove_index = rng.randrange(older_count)
        elif variant == 'b2':
            # 只刪最舊一個
            remove_index = 0
        modified_items = (
            real_items[:remove_index]
            + real_items[remove_index + 1:]
        )
        modified_seq = pad_or_truncate_sequence(
            modified_items,
            args.max_len,
        )
        output_sequences.append(modified_seq)
        changed_count += 1
    stats = {
        'batch_size': len(targets),
        'eligible_samples': eligible_count,
        'changed_samples': changed_count,
        'changed_ratio': round(
            changed_count / len(targets), 4
        ) if len(targets) else 0.0,
        'changed_ratio_among_eligible': round(
            changed_count / eligible_count, 4
        ) if eligible_count else None,
    }
    return output_sequences, stats

def calculate_hpr(seq, popular_items):
    history = [
        int(item)
        for item in seq
        if int(item) != 0
    ]
    if not history:
        return 0.0
    return (
        sum(item in popular_items for item in history)
        / len(history)
    )

def build_partial_tail_replay_data(
    train_data,
    unpopular_seen_items,
    replay_ratio,
    random_seed,
    popular_items,
    strategy='random',
):
    """
    保留全部原始資料。
    固定選取部分 Tail rows，各新增一份完整副本。
    不修改 seq、next 或 len_seq。
    """
    if not 0 <= replay_ratio <= 1:
        raise ValueError('replay_ratio must be between 0 and 1')
    original_data = train_data.copy().reset_index(drop=True)
    original_data['is_augmented'] = 0
    tail_mask = (
        original_data['next']
        .astype(int)
        .isin(unpopular_seen_items)
    )
    tail_data = original_data.loc[tail_mask]
    # 四捨五入後固定副本數量
    replay_count = int(len(tail_data) * replay_ratio + 0.5)
    # 每筆 Tail 樣本的歷史熱門比例
    tail_hprs = tail_data['seq'].apply(
        lambda seq: calculate_hpr(seq, popular_items)
    )
    if strategy == 'random':
        # 沿用目前 B4 的隨機選樣
        replay_data = tail_data.sample(
            n=replay_count,
            random_state=random_seed,
        ).copy()
    elif strategy == 'high_hpr':
        # 先隨機排列，讓同分樣本以固定 seed 決定順序
        shuffled_tail = tail_data.sample(
            frac=1,
            random_state=random_seed,
        )
        # 穩定排序：HPR 高者優先，同分保留前面的隨機順序
        ranked_hprs = tail_hprs.loc[
            shuffled_tail.index
        ].sort_values(
            ascending=False,
            kind='mergesort',
        )
        selected_indices = ranked_hprs.index[:replay_count]
        replay_data = tail_data.loc[
            selected_indices
        ].copy()
    else:
        raise ValueError(
            f'Unknown tail replay strategy: {strategy}'
        )
    selected_hprs = tail_hprs.loc[replay_data.index]
    # 複製 list，避免副本與原始 row 共用同一個 seq 物件
    replay_data['seq'] = replay_data['seq'].apply(list)
    replay_data['is_augmented'] = 1
    final_data = pd.concat(
        [original_data, replay_data],
        ignore_index=True,
    )
    # 不額外打亂；trainer 的 batch sampling 已會隨機抽樣
    original_tail_count = int(tail_mask.sum())
    final_tail_count = original_tail_count + replay_count
    stats = {
        'original_training_samples': len(original_data),
        'original_tail_samples': original_tail_count,
        'tail_replay_ratio': replay_ratio,
        'replay_copies': replay_count,
        'final_training_samples': len(final_data),
        'original_tail_ratio': round(
            original_tail_count / len(original_data), 4
        ),
        'final_tail_ratio': round(
            final_tail_count / len(final_data), 4
        ),
        'sequence_modified': False,
        'tail_replay_strategy': strategy,
        'all_tail_hpr_mean': (
            round(float(tail_hprs.mean()), 4)
            if len(tail_hprs) else None
        ),
        'selected_hpr_mean': (
            round(float(selected_hprs.mean()), 4)
            if len(selected_hprs) else None
        ),
        'selected_hpr_min': (
            round(float(selected_hprs.min()), 4)
            if len(selected_hprs) else None
        ),
        'selected_hpr_max': (
            round(float(selected_hprs.max()), 4)
            if len(selected_hprs) else None
        ),
    }
    return final_data, stats

def set_evaluation_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def model_train(train_data, val_data, test_data, con_data, model_joint, args, logger, pretrain_flag):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    eval_checkpoint = getattr(args, 'eval_checkpoint', None)
    # 只在 Target 階段啟用 evaluation-only
    if not pretrain_flag and eval_checkpoint is not None:
        model_joint.load_state_dict(
            torch.load(
                eval_checkpoint,
                map_location=device,
            )
        )
        # 後面的 epoch 迴圈不執行，直接進入 Test
        epochs = 0
        print("Evaluation only. Loaded:", eval_checkpoint)
        logger.info(
            "Evaluation only. Loaded: %s",
            eval_checkpoint,
        )
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0
    # 先保留一份初始模型，避免 best_model 尚未建立
    best_model = copy.deepcopy(model_joint)
    # A、B1、B2、B3 使用原始資料筆數；
    # 只有原始 B 會新增 Tail 副本。
    training_data_for_sampling = train_data
    unpopular_seen_items_for_augmentation = set()
    augmentation_stats = None
    sequence_variant = getattr(
        args, 'sequence_variant', 'none'
    )
    # 獨立亂數，不影響原本 batch sampling 的 seed
    sequence_rng = random.Random(args.random_seed + 1000)
    if not pretrain_flag:
        if (
            args.enable_tail_augmentation
            and sequence_variant != 'none'
        ):
            raise ValueError(
                'Tail duplication and B1/B2/B3 '
                'cannot be enabled together.'
            )
        # 必須從原始 training data 定義 Popular/Tail
        (
            popular_items_for_augmentation,
            unpopular_seen_items_for_augmentation,
            observed_train_items_for_augmentation,
            item_counter_for_augmentation,
        ) = build_item_popularity_groups(
            train_data=train_data,
            popular_ratio=args.popular_ratio,
        )
        if getattr(args, 'enable_tail_replay', False):
            # B4：部分 Tail 新增完整副本
            (
                training_data_for_sampling,
                augmentation_stats,
            ) = build_partial_tail_replay_data(
                train_data=train_data,
                unpopular_seen_items=(
                    unpopular_seen_items_for_augmentation
                ),
                replay_ratio=args.tail_replay_ratio,
                random_seed=args.random_seed,
                popular_items=popular_items_for_augmentation,
                strategy=getattr(
                    args,
                    'tail_replay_strategy',
                    'random',
                )
            )
            print("B4 Tail replay setup:", augmentation_stats)
            logger.info(
                "B4 Tail replay setup: %s",
                augmentation_stats,
            )
        elif args.enable_tail_augmentation:
            # 原始 B：這個分支內原有程式全部保留
            (
                training_data_for_sampling,
                augmentation_stats,
            ) = build_doubled_unpopular_seen_train_data(
                train_data=train_data,
                unpopular_seen_items=(
                    unpopular_seen_items_for_augmentation
                ),
                max_len=args.max_len,
                drop_probability=args.tail_drop_probability,
                preserve_recent=args.tail_preserve_recent,
                random_seed=args.random_seed,
            )
            print("Original B augmentation:", augmentation_stats)
            logger.info(
                "Original B augmentation: %s",
                augmentation_stats,
            )
        else:
            # A、B1、B2、B3 都不新增副本
            setup = {
                'sequence_variant': sequence_variant,
                'original_training_samples': len(train_data),
                'final_training_samples': len(
                    training_data_for_sampling
                ),
                'sequence_aug_prob': getattr(
                    args, 'sequence_aug_prob', 0.2
                ),
            }
            print("Sequence experiment setup:", setup)
            logger.info("Sequence experiment setup: %s", setup)
    num_rows = training_data_for_sampling.shape[0]
    num_batches = int(num_rows / args.batch_size)
    for epoch_temp in range(epochs):
        model_joint.train()
        flag_update = 0
        for j in range(num_batches):
            batch_seed = (
                args.random_seed
                + epoch_temp * num_batches
                + j
            )
            batch_df = training_data_for_sampling.sample(
            n=args.batch_size,
            random_state=batch_seed
            )
            if (
                not pretrain_flag
                and args.enable_tail_augmentation
                and epoch_temp == 0
                and j < 3
            ):
                batch_tail_count = (
                batch_df['next']
                .astype(int)
                .isin(unpopular_seen_items_for_augmentation)
                .sum()
                )
                batch_augmented_count = int(
                    batch_df['is_augmented'].sum()
                )
                batch_check = {
                    'batch_size': len(batch_df),
                    'tail_samples': int(batch_tail_count),
                    'augmented_samples': batch_augmented_count,
                    'tail_ratio': round(
                        batch_tail_count / len(batch_df),
                        4
                    ),
                    'augmented_ratio': round(
                        batch_augmented_count / len(batch_df),
                        4
                    )
                }
                print("Tail Dataset Batch Check:", batch_check)
                logger.info(
                    "Tail Dataset Batch Check: %s",
                    batch_check
                )
            batch = batch_df.to_dict()
            seq = list(batch['seq'].values())
            target = list(batch['next'].values())
            # 只修改 Target training 的輸入
            if (
                not pretrain_flag
                and sequence_variant in ('b1', 'b2', 'b3')
            ):
                seq, sequence_stats = perturb_sequence_batch(
                    sequences=seq,
                    targets=target,
                    unpopular_seen_items=(
                        unpopular_seen_items_for_augmentation
                    ),
                    args=args,
                    rng=sequence_rng,
                )
                if epoch_temp == 0 and j < 3:
                    print(
                        f"{sequence_variant.upper()} batch check:",
                        sequence_stats,
                    )
                    logger.info(
                        "%s batch check: %s",
                        sequence_variant.upper(),
                        sequence_stats,
                    )
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            target = (torch.LongTensor(target)).unsqueeze(1)
            seq = seq.to(device)
            target = target.to(device)
            if pretrain_flag:
                con_batch = con_data.sample(n=args.batch_size).to_dict()
                con_seq = list(con_batch['seq'].values())
                con_seq = torch.LongTensor(con_seq)
                con_seq = con_seq.to(device)
            else:
                con_seq = None
            scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis, em_loss = model_joint(seq, target, con_seq, pretrain_flag, args, epoch_temp, train_flag=True)  
            loss_diffu_value = model_joint.loss_diffu_ce(diffu_rep, target, pretrain_flag)  ## use this not above        
            loss_all = loss_diffu_value + em_loss*args.loss_lambda
            loss_all.backward()
            optimizer.step()
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        lr_scheduler.step()
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            with torch.no_grad():
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                # metrics_dict_mean = {}
                for j in range(int(val_data.shape[0]/args.batch_size)):
                    batch = val_data[j * args.batch_size: (j + 1)* args.batch_size].to_dict()
                    seq = list(batch['seq'].values())
                    target=list(batch['next'].values())
                    seq = torch.LongTensor(seq)
                    target = (torch.LongTensor(target)).unsqueeze(1)
                    seq = seq.to(device)
                    target = target.to(device)
                    scores_rec, rep_diffu, _, _, _, _ , _ = model_joint(seq, target, None, pretrain_flag, args, epoch_temp, train_flag=False)
                    scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu, pretrain_flag)    ### inner_production
                    output = torch.full_like(scores_rec_diffu, -100.0)
                    row_indices = torch.arange(args.batch_size).unsqueeze(1)
                    negative_samples = list(batch['negative_samples'].values())
                    output[row_indices, negative_samples] = scores_rec_diffu[row_indices, negative_samples]
                    scores_rec_diffu = output
                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, target, metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update += 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
            if flag_update == 0:
                bad_count += 1
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                if flag_update >= 3:
                    best_model = copy.deepcopy(model_joint)
                    if pretrain_flag:
                        torch.save(
                            model_joint.state_dict(),
                            args.source_model_path
                        )
                        print("Source model saved to:", args.source_model_path)
            if bad_count >= args.patience:
                break
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model_joint)
    # Test evaluation
    best_model.eval()
    eval_seed = getattr(args, 'eval_seed', 2026)
    set_evaluation_seed(eval_seed)
    print("Test evaluation seed:", eval_seed)
    logger.info("Test evaluation seed: %s", eval_seed)
    top_100_item = []
    with torch.no_grad():
        # 1. 初始化各組的指標加總
        overall_metric_sums = init_metric_sums(
            metric_ks
        )
        popular_metric_sums = init_metric_sums(
            metric_ks
        )
        unpopular_seen_metric_sums = init_metric_sums(
            metric_ks
        )
        unseen_metric_sums = init_metric_sums(
            metric_ks
        )
        overall_count = 0
        popular_count = 0
        unpopular_seen_count = 0
        unseen_count = 0
        # 計算推薦結果中有多少是 Popular item
        popular_recommend_count = {
            k: 0
            for k in metric_ks
        }
        total_recommend_count = {
            k: 0
            for k in metric_ks
        }
        popular_items = None
        unpopular_seen_items = None
        observed_train_items = None
        item_counter = None
        popular_lookup = None
        # 2. Target stage 才建立 Popular/Tail 分組
        if not pretrain_flag:
            popular_ratio = getattr(
                args,
                'popular_ratio',
                0.2
            )
            (
                popular_items,
                unpopular_seen_items,
                observed_train_items,
                item_counter
            ) = build_item_popularity_groups(
                train_data=train_data,
                popular_ratio=popular_ratio
            )
            popularity_definition = {
                'popular_ratio': popular_ratio,
                'observed_train_items': len(
                    observed_train_items
                ),
                'popular_items': len(
                    popular_items
                ),
                'unpopular_seen_items': len(
                    unpopular_seen_items
                )
            }
            print(
                "Popularity Definition"
                "------------------------------------------"
            )
            print(popularity_definition)
            logger.info(
                "Popularity Definition"
                "------------------------------------------"
            )
            logger.info(popularity_definition)
        # 3. 分 batch 測試
        for start_idx in range(
            0,
            test_data.shape[0],
            args.batch_size
        ):
            end_idx = min(
                start_idx + args.batch_size,
                test_data.shape[0]
            )
            batch_df = test_data.iloc[
                start_idx:end_idx
            ]
            current_batch_size = len(batch_df)
            seq = batch_df['seq'].tolist()
            target_list = (
                batch_df['next'].tolist()
            )
            negative_samples_list = (
                batch_df['negative_samples'].tolist()
            )
            seq = torch.LongTensor(
                seq
            ).to(device)
            target = (
                torch.LongTensor(target_list)
                .unsqueeze(1)
                .to(device)
            )
            # 4. 模型進行預測
            (
                scores_rec,
                rep_diffu,
                _,
                _,
                _,
                _,
                _
            ) = best_model(
                seq,
                target,
                None,
                pretrain_flag,
                args,
                0.2,
                train_flag=False
            )
            scores_rec_diffu = (
                best_model.diffu_rep_pre(
                    rep_diffu,
                    pretrain_flag
                )
            )
            # 5. 只保留 1 positive + 100 negatives
            output = torch.full_like(
                scores_rec_diffu,
                float('-inf')
            )
            row_indices = torch.arange(
                current_batch_size,
                device=device
            ).unsqueeze(1)
            negative_samples = torch.as_tensor(
                negative_samples_list,
                dtype=torch.long,
                device=device
            )
            output[
                row_indices,
                negative_samples
            ] = scores_rec_diffu[
                row_indices,
                negative_samples
            ]
            scores_rec_diffu = output
            # 6. 計算每筆 sample 的 HR/NDCG
            (
                sample_metrics,
                topk_indices
            ) = per_sample_metrics(
                scores_rec_diffu,
                target,
                metric_ks
            )
            overall_count += current_batch_size
            for (
                metric_name,
                metric_values
            ) in sample_metrics.items():
                overall_metric_sums[
                    metric_name
                ] += metric_values.sum().item()
            # 保留原始 Top-100 功能
            top_100_k = min(
                100,
                scores_rec_diffu.shape[1]
            )
            top_100_indices = torch.topk(
                scores_rec_diffu,
                k=top_100_k,
                dim=-1
            ).indices
            top_100_item.append(
                top_100_indices.detach().cpu()
            )
            # 7. Target stage：Popular/Tail 分組
            if not pretrain_flag:
                target_cpu = (
                    target.view(-1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                # Ground truth 是否為 Popular
                popular_mask = torch.tensor(
                    [
                        int(item) in popular_items
                        for item in target_cpu
                    ],
                    dtype=torch.bool
                )
                # Ground truth 是否曾在 train 中出現
                observed_mask = torch.tensor(
                    [
                        int(item) in observed_train_items
                        for item in target_cpu
                    ],
                    dtype=torch.bool
                )
                # 有出現過，但不是 Popular
                unpopular_seen_mask = (
                    observed_mask
                    & ~popular_mask
                )
                # 完全沒在 training data 中出現過
                unseen_mask = ~observed_mask
                batch_popular_count = int(
                    popular_mask.sum().item()
                )
                batch_unpopular_seen_count = int(
                    unpopular_seen_mask.sum().item()
                )
                batch_unseen_count = int(
                    unseen_mask.sum().item()
                )
                # 檢查三組是否完整涵蓋 batch
                batch_group_total = (
                    batch_popular_count
                    + batch_unpopular_seen_count
                    + batch_unseen_count
                )
                if (
                    batch_group_total
                    != current_batch_size
                ):
                    raise RuntimeError(
                        "Popularity grouping error: "
                        f"group total={batch_group_total}, "
                        f"batch size={current_batch_size}"
                    )
                popular_count += (
                    batch_popular_count
                )
                unpopular_seen_count += (
                    batch_unpopular_seen_count
                )
                unseen_count += (
                    batch_unseen_count
                )
                # 8. 將各筆指標加入對應群組
                for (
                    metric_name,
                    metric_values
                ) in sample_metrics.items():
                    if batch_popular_count > 0:
                        popular_metric_sums[
                            metric_name
                        ] += (
                            metric_values[
                                popular_mask
                            ]
                            .sum()
                            .item()
                        )
                    if (
                        batch_unpopular_seen_count
                        > 0
                    ):
                        unpopular_seen_metric_sums[
                            metric_name
                        ] += (
                            metric_values[
                                unpopular_seen_mask
                            ]
                            .sum()
                            .item()
                        )
                    if batch_unseen_count > 0:
                        unseen_metric_sums[
                            metric_name
                        ] += (
                            metric_values[
                                unseen_mask
                            ]
                            .sum()
                            .item()
                        )
                # 9. 建立 Popular item 查詢表
                if popular_lookup is None:
                    popular_lookup = torch.zeros(
                        scores_rec_diffu.shape[1],
                        dtype=torch.bool,
                        device=device
                    )
                    valid_popular_ids = [
                        int(item)
                        for item in popular_items
                        if (
                            0
                            <= int(item)
                            < popular_lookup.shape[0]
                        )
                    ]
                    if len(valid_popular_ids) > 0:
                        popular_lookup[
                            torch.LongTensor(
                                valid_popular_ids
                            ).to(device)
                        ] = True
                # 10. 計算推薦清單的 PopularRatio
                for k in metric_ks:
                    recommended_items = (
                        topk_indices[:, :k]
                    )
                    popular_recommend_count[
                        k
                    ] += (
                        popular_lookup[
                            recommended_items
                        ]
                        .sum()
                        .item()
                    )
                    total_recommend_count[
                        k
                    ] += (
                        current_batch_size * k
                    )
        # 11. 計算 Overall 平均結果
        test_metrics_dict_mean = (
            finalize_metric_sums(
                overall_metric_sums,
                overall_count
            )
        )
        print(
            "Test Overall"
            "------------------------------------------"
        )
        print(test_metrics_dict_mean)
        logger.info(
            "Test Overall"
            "------------------------------------------"
        )
        logger.info(test_metrics_dict_mean)
        # 12. 顯示 Popular/Tail 結果
        if not pretrain_flag:
            popular_metrics_dict = (
                finalize_metric_sums(
                    popular_metric_sums,
                    popular_count
                )
            )
            unpopular_seen_metrics_dict = (
                finalize_metric_sums(
                    unpopular_seen_metric_sums,
                    unpopular_seen_count
                )
            )
            unseen_metrics_dict = (
                finalize_metric_sums(
                    unseen_metric_sums,
                    unseen_count
                )
            )
            popular_ratio_dict = {
                f'PopularRatio@{k}': round(
                    popular_recommend_count[k]
                    / total_recommend_count[k]
                    * 100,
                    4
                )
                if total_recommend_count[k] > 0
                else None
                for k in metric_ks
            }
            group_size_dict = {
                'Overall_test_samples': (
                    overall_count
                ),
                'Popular_test_samples': (
                    popular_count
                ),
                'Unpopular_seen_test_samples': (
                    unpopular_seen_count
                ),
                'Unseen_test_samples': (
                    unseen_count
                ),
                'Grouped_test_samples': (
                    popular_count
                    + unpopular_seen_count
                    + unseen_count
                )
            }
            print(
                "Test Group Size"
                "------------------------------------------"
            )
            print(group_size_dict)
            print(
                "Test Popular Ground Truth"
                "------------------------------------------"
            )
            print(popular_metrics_dict)
            print(
                "Test Unpopular-Seen Ground Truth"
                "------------------------------------------"
            )
            print(unpopular_seen_metrics_dict)
            print(
                "Test Unseen Ground Truth"
                "------------------------------------------"
            )
            print(unseen_metrics_dict)
            print(
                "Recommendation Popularity"
                "------------------------------------------"
            )
            print(popular_ratio_dict)
            logger.info(group_size_dict)
            logger.info(popular_metrics_dict)
            logger.info(
                unpopular_seen_metrics_dict
            )
            logger.info(unseen_metrics_dict)
            logger.info(popular_ratio_dict)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    print(args)
    return best_model, test_metrics_dict_mean