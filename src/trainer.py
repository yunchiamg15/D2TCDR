"""One training/evaluation loop. The method only controls added training copies."""
import csv
import json
import math
from pathlib import Path

import pandas as pd
import torch

from protocol import digest_file, digest_json, isolated_rng, seed_all
from agent.b3 import build_b3_candidates
from agent.policy import AugmentationController, LEARNED, empty_stats
from agent.reward import calculate_reward

PHASE_OFFSETS = {'val': 100, 'res': 200, 'test': 300}


def evaluate(model, data, args, popular, tail, phase):
    if phase not in PHASE_OFFSETS:
        raise ValueError('Unknown evaluation phase')
    ks = args.metric_ks
    names = ('Overall', 'Popular', 'Tail', 'Unseen')
    counts = {name: 0 for name in names}
    sums = {name: {f'{metric}@{k}': 0. for metric in ('HR','NDCG') for k in ks} for name in names}
    popular_recs = {k: 0 for k in ks}
    previous_mode = model.training
    try:
        with isolated_rng(args.eval_seed + PHASE_OFFSETS[phase]):
            model.eval()
            with torch.no_grad():
                for start in range(0, len(data), args.batch_size):
                    batch = data.iloc[start:start + args.batch_size]
                    seq = torch.tensor(batch['seq'].tolist(), dtype=torch.long, device=args.device)
                    targets = torch.tensor(batch['next'].tolist(), dtype=torch.long, device=args.device).view(-1,1)
                    candidates = torch.tensor(batch['negative_samples'].tolist(), dtype=torch.long, device=args.device)
                    if not torch.equal(candidates[:,0], targets.view(-1)):
                        raise ValueError('Positive must be candidate 0')
                    _, rep, _, _, _, _, _ = model(seq, targets, None, False, args, 0, train_flag=False)
                    scores = model.diffu_rep_pre(rep, False).gather(1, candidates)
                    ranked = scores.topk(max(ks), dim=1).indices.cpu()
                    ranked_items = candidates.gather(1, ranked.to(candidates.device)).cpu().tolist()
                    hits = ranked.eq(0)
                    labels = targets.view(-1).cpu().tolist()
                    masks = dict(Overall=torch.ones(len(batch), dtype=torch.bool),
                        Popular=torch.tensor([v in popular for v in labels]),
                        Tail=torch.tensor([v in tail for v in labels]),
                        Unseen=torch.tensor([v not in popular and v not in tail for v in labels]))
                    for name in names:
                        counts[name] += int(masks[name].sum())
                    for k in ks:
                        hr = hits[:,:k].any(1).double()
                        discounts = 1 / torch.log2(torch.arange(2,k+2,dtype=torch.float64))
                        ndcg = (hits[:,:k].double() * discounts).sum(1)
                        for name in names:
                            sums[name][f'HR@{k}'] += float(hr[masks[name]].sum())
                            sums[name][f'NDCG@{k}'] += float(ndcg[masks[name]].sum())
                        popular_recs[k] += sum(v in popular for row in ranked_items for v in row[:k])
    finally:
        model.train(previous_mode)
    metrics = {name: {key: value / counts[name] if counts[name] else None
                      for key,value in sums[name].items()} for name in names}
    recommendation = {f'PopularRatio@{k}': popular_recs[k] / (counts['Overall'] * k)
                      if counts['Overall'] else None for k in ks}
    return dict(metrics=metrics, counts=counts, recommendation_popularity=recommendation)


def reward_metrics(evaluation):
    return {
        'Overall_HR@10': evaluation['metrics']['Overall']['HR@10'],
        'Popular_HR@10': evaluation['metrics']['Popular']['HR@10'],
        'Tail_HR@10': evaluation['metrics']['Tail']['HR@10'],
        'counts': evaluation['counts']}


def load_reference(path, fingerprint):
    reference = json.loads(Path(path).read_text(encoding='utf-8'))
    if reference.get('method') != 'baseline' or reference.get('protocol_fingerprint') != fingerprint:
        raise ValueError('Use the Shared_Baseline reference from this same shared protocol; old V3/B3 references are incompatible')
    metrics = reference['metrics']
    for key in ('Overall_HR@10','Popular_HR@10','Tail_HR@10'):
        value = metrics[key]
        if not isinstance(value, (int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Reference metrics must be finite ratios in [0,1]')
    return metrics



def write_comparison(directory, fingerprint):
    methods = ['baseline','random_b3','v3_selection','copy_replace','four_head']
    records = {}
    for path in directory.glob('*_results.json'):
        result = json.loads(path.read_text(encoding='utf-8'))
        meta = result['metadata']
        if meta['protocol_fingerprint'] == fingerprint:
            records[meta['method']] = result
    if 'baseline' not in records:
        return
    rows = []
    baseline = records['baseline']['test']
    for group in ('Overall','Popular','Tail'):
        for metric in baseline['metrics'][group]:
            reference = baseline['metrics'][group][metric]
            row = [f'{group} {metric}']
            for method in methods:
                value = records.get(method,{}).get('test',{}).get('metrics',{}).get(group,{}).get(metric)
                if value is None:
                    row.append('')
                elif method == 'baseline':
                    row.append(f'{value*100:.4f}')
                elif reference:
                    row.append(f'{value*100:.4f} ({(value/reference-1)*100:+.2f}%)')
                else:
                    row.append(f'{value*100:.4f} (N/A)')
            rows.append(row)
    path = directory/'Shared_comparison.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['Metric','Baseline','Random B3','V3 Selection Agent','CopyReplace Agent','Four Head Agent'])
        writer.writerows(rows)


def train_method(model, train, val, res, test, popular, tail, protocol, args):
    directory = Path(args.output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    prefix = directory / args.model_name
    target_path = Path(str(prefix) + '_target.pth')
    policy_path = Path(str(prefix) + '_target_policy.pth')
    log_path = Path(str(prefix) + '_training.jsonl')
    metadata_path = Path(str(prefix) + '_metadata.json')
    result_path = Path(str(prefix) + '_results.json')
    learned = args.method in LEARNED
    references = load_reference(args.reward_reference, protocol['fingerprint']) if learned else None
    controller = None
    if args.method != 'baseline':
        candidates = build_b3_candidates(train, args.min_cooccurrence)
        controller = AugmentationController(train,popular,tail,candidates,args)
    # Agent construction and candidate building cannot change model training RNG.
    seed_all(args.target_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.decay_step,args.gamma)
    batches = len(train) // args.batch_size
    best_score = float('-inf')
    best_state = None
    best_epoch = -1
    bad_count = 0
    total_actions = {key:0 for key in ('tail_samples','copies','replaced','full_copies')}
    with log_path.open('w',encoding='utf-8') as log:
        log.write(json.dumps(dict(event='start',method=args.method,protocol_fingerprint=protocol['fingerprint'],
            train_samples=len(train),res_samples=len(res),validation_samples=len(val),test_samples=len(test),
            popular_items=len(popular),tail_items=len(tail),batches_per_epoch=batches,
            eligibility='all_tail_next',reference=references,
            method_config=dict(agent_budget_ratio=args.agent_budget_ratio,
                agent_lr=args.agent_lr,agent_hidden=args.agent_hidden,
                agent_entropy_coef=args.agent_entropy_coef,
                reward_lambda_overall=args.reward_lambda_overall,
                reward_lambda_popular=args.reward_lambda_popular,
                reward_epsilon_overall=args.reward_epsilon_overall,
                reward_epsilon_popular=args.reward_epsilon_popular)))+'\n')
        for epoch in range(args.epochs):
            model.train()
            shuffled = train.sample(frac=1,random_state=args.random_seed+epoch).reset_index(drop=True)
            epoch_totals = {key:0 for key in total_actions}
            for j in range(batches):
                original = shuffled.iloc[j*args.batch_size:(j+1)*args.batch_size].copy().reset_index(drop=True)
                original_hash = digest_json([int(v) for v in original['_row_id']])
                lp = entropy = None
                if controller:
                    batch,lp,entropy,actions = controller.augment(original,epoch)
                else:
                    batch = original
                    actions = empty_stats()
                    actions['tail_samples'] = sum(int(y) in tail for y in original['next'])
                    actions['skipped'] = actions['tail_samples']
                seq=torch.tensor(batch['seq'].tolist(),dtype=torch.long,device=args.device)
                target=torch.tensor(batch['next'].tolist(),dtype=torch.long,device=args.device).view(-1,1)
                optimizer.zero_grad()
                _,rep,_,_,_,_,auxiliary=model(seq,target,None,False,args,epoch,train_flag=True)
                # No method has a per-sample weight, alpha or Weight Head.
                loss=model.loss_diffu_ce(rep,target,False)+auxiliary*args.loss_lambda
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite recommendation loss')
                loss.backward()
                optimizer.step()
                res_values = update = None
                if learned and lp is not None:
                    res_values = reward_metrics(evaluate(model,res,args,popular,tail,'res'))
                    update = controller.update(calculate_reward(res_values,references,args),lp,entropy)
                record=dict(event='batch',epoch=epoch,batch=j,original_batch_hash=original_hash,
                    original_batch_size=len(original),training_batch_size=len(batch),
                    loss=float(loss.detach()),actions=actions,res_metrics=res_values,update=update)
                log.write(json.dumps(record)+'\n')
                for key in total_actions:
                    epoch_totals[key] += actions[key]
                    total_actions[key] += actions[key]
            scheduler.step()
            print('Epoch summary:',dict(epoch=epoch,**epoch_totals))
            if (epoch+1)%args.eval_interval==0 or epoch+1==args.epochs:
                evaluation = evaluate(model,val,args,popular,tail,'val')
                score = evaluation['metrics']['Overall']['NDCG@10']
                if score is None or not math.isfinite(score):
                    raise ValueError('Invalid Validation NDCG@10')
                improved = score > best_score
                if improved:
                    best_score,best_epoch,bad_count=score,epoch,0
                    best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                    torch.save(best_state,target_path)
                    if learned:
                        torch.save(dict(method=args.method,epoch=epoch,protocol_fingerprint=protocol['fingerprint'],
                            state_dict={k:v.detach().cpu().clone() for k,v in controller.policy.state_dict().items()}),policy_path)
                else:
                    bad_count += 1
                log.write(json.dumps(dict(event='validation',epoch=epoch,score=score,
                    improved=improved,metrics=evaluation))+'\n')
                log.flush()
                print('Validation:',dict(epoch=epoch,NDCG10=round(score*100,4),best_epoch=best_epoch,
                                         best_NDCG10=round(best_score*100,4),bad_count=bad_count))
                if bad_count>=args.patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    res_evaluation=evaluate(model,res,args,popular,tail,'res')
    test_evaluation=evaluate(model,test,args,popular,tail,'test')
    metadata=dict(method=args.method,model_name=args.model_name,protocol_fingerprint=protocol['fingerprint'],
        source_sha256=protocol['inputs']['source_sha256'],selected_epoch=best_epoch,
        selection_rule='Overall Validation NDCG@10',validation_ndcg10=best_score,
        counts=dict(main=len(train),res=len(res),validation=len(val),test=len(test)),
        total_actions=total_actions,common_config=protocol['config'],
        method_code_sha256=digest_file(Path(__file__).parent/'agent'/'policy.py'),
        method_config=dict(agent_budget_ratio=args.agent_budget_ratio,agent_lr=args.agent_lr,
            agent_hidden=args.agent_hidden,agent_entropy_coef=args.agent_entropy_coef,
            reward_baseline_window=args.reward_baseline_window,
            reward_lambda_overall=args.reward_lambda_overall,reward_lambda_popular=args.reward_lambda_popular,
            reward_epsilon_overall=args.reward_epsilon_overall,reward_epsilon_popular=args.reward_epsilon_popular))
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    result=dict(metadata=metadata,res=res_evaluation,test=test_evaluation)
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.method=='baseline':
        reference_path=Path(str(prefix)+'_target_reward_reference.json')
        reference_path.write_text(json.dumps(dict(method='baseline',protocol_fingerprint=protocol['fingerprint'],
            metrics=reward_metrics(res_evaluation),target_checkpoint=str(target_path),
            selected_epoch=best_epoch),indent=2),encoding='utf-8')
        print('Reward reference saved:',reference_path)
    print('Final selected checkpoint:',dict(epoch=best_epoch,val_ndcg10=best_score*100))
    print('REWARD_SET_RESULTS:',reward_metrics(res_evaluation))
    print('Test Group Size:',test_evaluation['counts'])
    for name,values in test_evaluation['metrics'].items():
        print('Test '+name+':',{key:round(value*100,4) if value is not None else None for key,value in values.items()})
    print('Recommendation Popularity:',{key:round(value*100,4) for key,value in test_evaluation['recommendation_popularity'].items()})
    print('Target model saved:',target_path)
    write_comparison(directory,protocol['fingerprint'])
    print('Results saved:',result_path)
    print('Comparison table:',directory/'Shared_comparison.csv')
    return result
