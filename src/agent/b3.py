from collections import Counter


def build_b3_candidates(train_data, min_cooccurrence=2):
    """
    從未擴增的 Target Training histories 建立共現候選。

    c(i)   = 包含 item i 的 histories 數量
    c(i,j) = 同時包含 i、j 的 histories 數量
    s(i,j) = c(i,j) / sqrt(c(i)*c(j))
    """
    item_counts = Counter()
    pair_counts = Counter()

    for seq in train_data["seq"]:
        # 每筆歷史先去重，排除 padding
        items = sorted({
            int(item) for item in seq
            if int(item) != 0
        })

        item_counts.update(items)

        for p, left in enumerate(items):
            for right in items[p + 1:]:
                pair_counts[(left, right)] += 1

    candidates = {}

    for (left, right), count in pair_counts.items():
        if count < min_cooccurrence:
            continue

        similarity = count / (
            (item_counts[left] * item_counts[right]) ** 0.5
        )

        candidates.setdefault(left, []).append(
            (right, similarity)
        )
        candidates.setdefault(right, []).append(
            (left, similarity)
        )

    # 相似度由高到低；同分以 Item ID 排序
    for item in candidates:
        candidates[item].sort(
            key=lambda pair: (-pair[1], pair[0])
        )

    return candidates


def augment_selected_b3(
    duplicate_rows,
    popular_items,
    candidates,
    rng,
    preserve_recent=2,
    replacement_topk=10
):
    """
    B3：只修改 Agent 選中的副本。

    - 隨機選一個較早的 Popular Item
    - 保留最近兩個有效 Items
    - 排除 padding / history / next
    - 從合法的前 10 個相關候選等機率抽取
    - 無合法候選時，副本維持原樣
    - 不修改 len_seq、next 和原始樣本
    """
    rows = duplicate_rows.copy().reset_index(drop=True)

    # 確保副本的 seq 與原始資料不共用 list
    rows["seq"] = rows["seq"].map(
        lambda seq: list(seq)
    )

    stats = {
        "copies": len(rows),
        "replaced": 0,
        "too_short": 0,
        "no_older_popular": 0,
        "no_candidate": 0
    }

    for row_id in rows.index:
        original_seq = rows.at[row_id, "seq"]

        valid_positions = [
            pos
            for pos, item in enumerate(original_seq)
            if int(item) != 0
        ]

        # 最近兩個有效 Items 不能修改
        if len(valid_positions) <= preserve_recent:
            stats["too_short"] += 1
            continue

        # 只能選較早的 Popular Item
        older_popular_positions = [
            pos
            for pos in valid_positions[:-preserve_recent]
            if int(original_seq[pos]) in popular_items
        ]

        if not older_popular_positions:
            stats["no_older_popular"] += 1
            continue

        # 隨機選一個可替換位置
        replace_pos = rng.choice(
            older_popular_positions
        )

        old_item = int(original_seq[replace_pos])

        # 不得替換成 history 已有、next 或 padding
        blocked = {
            int(item)
            for item in original_seq
        }
        blocked.add(0)
        blocked.add(int(rows.at[row_id, "next"]))

        # 先過濾，再取相似度最高的前 10 個
        top_candidates = [
            item
            for item, _ in candidates.get(old_item, [])
            if item not in blocked
        ][:replacement_topk]

        if not top_candidates:
            stats["no_candidate"] += 1
            continue

        # 修改副本中的一個 Item
        augmented_seq = list(original_seq)
        augmented_seq[replace_pos] = rng.choice(
            top_candidates
        )

        rows.at[row_id, "seq"] = augmented_seq
        stats["replaced"] += 1

    return rows, stats