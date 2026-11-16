import os
import hashlib
import json
from pathlib import Path
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train
from agent.high_hpr_policy import LEARNED
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--s_dataset', default='movie')
parser.add_argument('--t_dataset', default='video')
parser.add_argument('--p', type=float, default=0.3)
parser.add_argument('--w', type=float, default=1.0)
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=1997, help='Random seed')  
parser.add_argument('--max_len', type=int, default=8, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')  
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training') 
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--popular_ratio', type=float, default=0.2, help='Top ratio of target-domain training items treated as popular')
# ==============================
# RL Agent Settings
# ==============================
parser.add_argument('--method', choices=[
    'baseline', 'model_b', 'b3_full', 'random_b3',
    'v3_selection', 'copy_replace', 'four_head', 'four_head_embedding',
    'four_head_embedding_top10',
], default=None, help='Training method (default: four_head_embedding)')
parser.add_argument('--high_hpr_pool_ratio', type=float, default=0.20)
parser.add_argument('--preserve_recent', type=int, default=2)
parser.add_argument('--replacement_topk', type=int, default=10)
parser.add_argument('--min_cooccurrence', type=int, default=2)
parser.add_argument('--agent_entropy_coef', type=float, default=0.0)
parser.add_argument('--agent_enabled', type=int, default=0, choices=[0, 1])
parser.add_argument('--agent_lr', type=float, default=0.0001)
parser.add_argument('--agent_hidden', type=int, default=32)
parser.add_argument('--agent_budget_ratio', type=float, default=0.20)
# ==============================
# Reward Settings
# ==============================
parser.add_argument('--reward_holdout_ratio', type=float, default=0.10)
parser.add_argument('--reward_eval_samples', type=int, default=256)
parser.add_argument('--reward_ref_overall', type=float, default=None)
parser.add_argument('--reward_ref_popular', type=float, default=None)
parser.add_argument('--reward_lambda_overall', type=float, default=1.0)
parser.add_argument('--reward_lambda_popular', type=float, default=1.0)
parser.add_argument('--reward_epsilon_overall', type=float, default=0.002)
parser.add_argument('--reward_epsilon_popular', type=float, default=0.002)
parser.add_argument('--reward_baseline_window', type=int, default=50)
parser.add_argument('--skip_source_pretrain', action='store_true')
parser.add_argument('--init_source_model', default=None)
parser.add_argument('--model_name', default=None)
parser.add_argument('--reward_reference', default=None,
                    help='JSON from a matching V3_NoAgent run; overrides numeric references')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_lambda', type=float, default=1, help='loss weight for diffusion')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--diffusion_steps', type=int, default=32, help='Diffusion step')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
parser.add_argument('--eval_interval', type=int, default=5, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=2, help='the number of epoch to wait before early stop')
parser.add_argument('--description', type=str, default='Diffu_norm_score', help='Model brief introduction')
args = parser.parse_args()
if args.method is None:
    args.method = 'four_head_embedding'
    args.skip_source_pretrain = True
    if args.init_source_model is None:
        args.init_source_model = '/mnt/mira/D2TCDR/src/saved_model/movie_video/traditional_source.pth'
    if args.reward_reference is None and args.reward_ref_overall is None and args.reward_ref_popular is None:
        args.reward_reference = str(Path('saved_model') / f'{args.s_dataset}_{args.t_dataset}'
                                    / 'V3_NoAgent_RES90_target_reward_reference.json')
    if args.model_name is None:
        args.model_name = 'HighHPR_FourHeadEmbedding_Logged'
# --method is authoritative; all learned methods have no Loss Weight Head.
args.agent_enabled = int(args.method in LEARNED)
if args.model_name is None:
    args.model_name = {
        'baseline': 'V3_NoAgent_RES90',
        'model_b': 'HighHPR_ModelB_Full',
        'b3_full': 'HighHPR_B3_Full',
        'random_b3': 'HighHPR_RandomB3_Budget20',
        'v3_selection': 'HighHPR_V3Selection',
        'copy_replace': 'HighHPR_CopyReplace',
        'four_head': 'HighHPR_FourHead',
        'four_head_embedding': 'HighHPR_FourHeadEmbedding',
        'four_head_embedding_top10': 'HighHPR_FourHeadEmbeddingTop10',
    }[args.method]
args.target_seed = args.random_seed + 1
args.augmentation_seed = args.random_seed + 2000
if not 0 < args.high_hpr_pool_ratio <= 1:
    parser.error('--high_hpr_pool_ratio must be in (0, 1]')
if args.preserve_recent < 2 or args.replacement_topk < 1 or args.min_cooccurrence < 1:
    parser.error('Invalid replacement settings')
if args.num_gpu != 1:
    parser.error('This implementation supports num_gpu=1')
if not 0 <= args.agent_budget_ratio <= 1:
    parser.error('--agent_budget_ratio must be in [0, 1]')
if args.reward_eval_samples < 0 or args.reward_baseline_window < 1:
    parser.error('Invalid Reward evaluation size / baseline window')
if not args.model_name or Path(args.model_name).name != args.model_name or args.model_name in ('.','..'):
    parser.error('--model_name must be a filename, without directory components')


if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.s_dataset + "_" + args.t_dataset):
    os.makedirs(args.log_file + args.s_dataset + "_" + args.t_dataset )


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.s_dataset + "_" + args.t_dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, source_item_num, target_item_num):
    args.source_item_num = source_item_num
    args.target_item_num = target_item_num
    args.item_num = source_item_num
    args.pretrain_flag = True
    return args

def main(args):    
    fix_random_seed_as(args.random_seed)
    if args.t_dataset == "toy" or args.t_dataset == "sports":
        args.dropout = 0.3
    print(args)
    logger.info(args)
    path_s_data = '../data/' + args.s_dataset
    s_train_path = path_s_data + '/train.df'
    s_val_path = path_s_data + '/val.df'
    s_test_path = path_s_data + '/test.df'
    path_t_data = '../data/' + args.t_dataset
    t_train_path = path_t_data + '/train.df'
    t_val_path = path_t_data + '/val.df'
    t_test_path = path_t_data + '/test.df'

    
    s_tra_data = pd.read_pickle(s_train_path)
    s_val_data = pd.read_pickle(s_val_path)
    s_test_data = pd.read_pickle(s_test_path)
    t_tra_data = pd.read_pickle(t_train_path)
    t_val_data = pd.read_pickle(t_val_path)
    t_test_data = pd.read_pickle(t_test_path)

    # ==============================
    # Reward Holdout Split
    # ==============================

    if args.agent_enabled:
        if (
            not args.reward_reference and (
                args.reward_ref_overall is None
                or args.reward_ref_popular is None)
        ):
            raise ValueError(
                "Agent requires reward baseline references."
            )

        if args.num_gpu != 1:
            raise ValueError(
                "Agent prototype currently supports one GPU."
            )

    if not 0 < args.reward_holdout_ratio < 1:
        raise ValueError("Invalid reward holdout ratio.")

    # Fixed and reproducible split
    reward_indices = t_tra_data.sample(
        frac=args.reward_holdout_ratio,
        random_state=args.random_seed
    ).index

    t_reward_data = (
        t_tra_data.loc[reward_indices]
        .copy()
        .reset_index(drop=True)
    )

    t_tra_data = (
        t_tra_data.drop(index=reward_indices)
        .reset_index(drop=True)
    )

    print("Target Train:", len(t_tra_data))
    print("Reward Set:", len(t_reward_data))

    source_item_num = 98507
    target_item_num = 23978

    args = item_num_create(args, source_item_num, target_item_num)
    
    s_val_data['negative_samples'] = None
    m = s_val_data['seq'].size
    for i in range(m):
        negative_sample = []
        negative_sample.append(s_val_data['next'][i])
        while len(negative_sample)<101 :
            sample = random.randint(0, args.source_item_num - 1)
            if sample not in negative_sample:
                negative_sample.append(sample)
        s_val_data.at[i, 'negative_samples'] = negative_sample
    t_val_data['negative_samples'] = None
    m = t_val_data['seq'].size
    for i in range(m):
        negative_sample = []
        negative_sample.append(t_val_data['next'][i])
        while len(negative_sample)<101 :
            sample = random.randint(0, args.target_item_num - 1)
            if sample not in negative_sample:
                negative_sample.append(sample)
        t_val_data.at[i, 'negative_samples'] = negative_sample
    # ==============================
    # Fixed Reward Candidates
    # ==============================

    reward_rng = random.Random(args.random_seed + 303)

    t_reward_data['negative_samples'] = None

    for i in range(len(t_reward_data)):
        positive = int(t_reward_data.at[i, 'next'])

        candidates = [positive]

        while len(candidates) < 101:
            candidate = reward_rng.randint(
                1, args.target_item_num
            )

            if candidate not in candidates:
                candidates.append(candidate)

        t_reward_data.at[
            i, 'negative_samples'
        ] = candidates
    s_test_data['negative_samples'] = None
    m = s_test_data['seq'].size
    for i in range(m):
        negative_sample = []
        negative_sample.append(s_test_data['next'][i])
        while len(negative_sample)<101 :
            sample = random.randint(0, args.source_item_num - 1)
            if sample not in negative_sample:
                negative_sample.append(sample)
        s_test_data.at[i, 'negative_samples'] = negative_sample
    t_test_data['negative_samples'] = None
    m = t_test_data['seq'].size
    for i in range(m):
        negative_sample = []
        negative_sample.append(t_test_data['next'][i])
        while len(negative_sample)<101 :
            sample = random.randint(0, args.target_item_num - 1)
            if sample not in negative_sample:
                negative_sample.append(sample)
        t_test_data.at[i, 'negative_samples'] = negative_sample
        
    diffu_rec = create_model_diffu(args)
    rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)
    
    # ==============================
    # Stage 1: Source Pretraining
    # ==============================

    model_dir = (
        "./saved_model/"
        + args.s_dataset + "_"
        + args.t_dataset
    )

    os.makedirs(model_dir, exist_ok=True)

    args.source_model_path = model_dir + '/' + args.model_name + '_source.pth'
    args.target_model_path = model_dir + '/' + args.model_name + '_target.pth'
    source_checkpoint = (args.init_source_model if args.skip_source_pretrain and args.init_source_model
                         else args.source_model_path)

    pretrain_flag = True

    if not args.skip_source_pretrain:
        best_model, _ = model_train(
            s_tra_data,
            s_val_data,
            s_test_data,
            t_tra_data,
            rec_diffu_joint_model,
            args,
            logger,
            pretrain_flag
        )

        torch.save(
            best_model.state_dict(),
            source_checkpoint
        )

    # Load fixed source checkpoint
    rec_diffu_joint_model.load_state_dict(
        torch.load(
            source_checkpoint,
            map_location=args.device
        )
    )

    # ==============================
    # Stage 2: Target Training
    # ==============================

    # Reference compatibility: original V3 split/candidates and identical Source.
    digest = hashlib.sha256()
    config_keys = ('s_dataset','t_dataset','random_seed','reward_holdout_ratio',
        'reward_eval_samples','popular_ratio','max_len','batch_size','hidden_size',
        'num_blocks','dropout','emb_dropout','p','w','diffusion_steps',
        'schedule_sampler_name','noise_schedule','rescale_timesteps','lambda_uncertainty')
    config = {key: getattr(args,key) for key in config_keys}
    source_digest = hashlib.sha256()
    with open(source_checkpoint, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            source_digest.update(block)
    config['source_sha256'] = source_digest.hexdigest()
    config['protocol'] = 'github_agent_v3_215cba4'
    digest.update(json.dumps(config,sort_keys=True).encode())
    for data in (t_tra_data, t_reward_data, t_val_data):
        for _, row in data.iterrows():
            record = [[int(v) for v in row['seq']], int(row['next'])]
            if 'negative_samples' in data:
                record.append([int(v) for v in row['negative_samples']])
            digest.update(json.dumps(record,separators=(',',':')).encode())
    args.reward_fingerprint = digest.hexdigest()
    if args.agent_enabled and args.reward_reference:
        reference = json.loads(Path(args.reward_reference).read_text(encoding='utf-8'))
        if reference.get('method') != 'V3_NoAgent' or reference.get('fingerprint') != args.reward_fingerprint:
            raise ValueError('Use a matching V3_NoAgent reference; older B3 references use a different protocol')
        args.reward_ref_overall = reference['metrics']['Overall_HR@10']
        args.reward_ref_popular = reference['metrics']['Popular_HR@10']

    args.item_num = target_item_num
    args.eval_interval = 1
    args.patience = 5

    pretrain_flag = False
    # Fixed Target Training Random Seed
    fix_random_seed_as(args.random_seed + 1)
    best_model, test_results = model_train(
        t_tra_data,
        t_val_data,
        t_test_data,
        None,
        rec_diffu_joint_model,
        args,
        logger,
        pretrain_flag,
        reward_data=t_reward_data
    )

    target_model_path = args.target_model_path

    torch.save(best_model.state_dict(), target_model_path)

    print("Target model saved at:", target_model_path)
    logger.info("Target model saved at: %s", target_model_path)

if __name__ == '__main__':
    main(args)
