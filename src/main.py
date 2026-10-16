import os
import random
import argparse
import logging
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd

from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()

# Dataset / model
parser.add_argument('--s_dataset', default='movie')
parser.add_argument('--t_dataset', default='video')
parser.add_argument('--p', type=float, default=0.3)
parser.add_argument('--w', type=float, default=1.0)
parser.add_argument('--log_file', default='log/')
parser.add_argument('--random_seed', type=int, default=1997)
parser.add_argument('--max_len', type=int, default=8)
parser.add_argument(
    '--device', default='cuda', choices=['cpu', 'cuda']
)
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--emb_dropout', type=float, default=0.3)
parser.add_argument('--hidden_act', default='gelu')
parser.add_argument('--num_blocks', type=int, default=4)

# Training
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument(
    '--metric_ks', nargs='+', type=int, default=[5, 10, 20]
)
parser.add_argument(
    '--optimizer', default='Adam', choices=['SGD', 'Adam']
)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--loss_lambda', type=float, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--schedule_sampler_name', default='lossaware')
parser.add_argument('--diffusion_steps', type=int, default=32)
parser.add_argument('--lambda_uncertainty', type=float, default=0.001)
parser.add_argument('--noise_schedule', default='trunc_lin')
parser.add_argument('--rescale_timesteps', default=True)
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--description', default='Diffu_norm_score')

# Popularity definition
parser.add_argument(
    '--popular_ratio',
    type=float,
    default=0.2,
)

# Checkpoints
parser.add_argument('--model_name', default='traditional')
parser.add_argument('--skip_source_pretrain', action='store_true')
parser.add_argument('--init_source_model', default=None)
parser.add_argument('--eval_checkpoint', default=None)
parser.add_argument('--eval_seed', type=int, default=2026)

# Tail replay: random / high HPR
parser.add_argument('--enable_tail_replay', action='store_true')
parser.add_argument(
    '--tail_replay_ratio',
    type=float,
    default=0.2,
    help='Fraction of Tail training samples duplicated',
)
parser.add_argument(
    '--tail_replay_strategy',
    default='random',
    choices=['random', 'high_hpr'],
    help='Tail replay selection: random or high_hpr',
)

# Retained because the current trainer.py references these attributes
parser.add_argument('--enable_tail_augmentation', action='store_true')
parser.add_argument('--tail_drop_probability', type=float, default=0.2)
parser.add_argument('--tail_preserve_recent', type=int, default=2)
parser.add_argument(
    '--sequence_variant',
    default='none',
    choices=['none', 'b1', 'b2', 'b3'],
)
parser.add_argument('--sequence_aug_prob', type=float, default=0.2)
parser.add_argument('--sequence_preserve_recent', type=int, default=2)

args = parser.parse_args()

if not 0 <= args.tail_replay_ratio <= 1:
    parser.error('--tail_replay_ratio must be between 0 and 1')

if args.enable_tail_replay and (
    args.enable_tail_augmentation
    or args.sequence_variant != 'none'
):
    parser.error(
        'Tail replay cannot be combined with '
        'original B augmentation or B1/B2/B3'
    )

if args.enable_tail_augmentation and args.sequence_variant != 'none':
    parser.error(
        '--enable_tail_augmentation cannot be combined with '
        '--sequence_variant b1/b2/b3'
    )

if not 0 <= args.sequence_aug_prob <= 1:
    parser.error('--sequence_aug_prob must be between 0 and 1')

if args.sequence_preserve_recent < 2:
    parser.error('--sequence_preserve_recent must be at least 2')

log_dir = os.path.join(
    args.log_file,
    args.s_dataset + "_" + args.t_dataset,
)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(
        log_dir,
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
    ),
    datefmt='%Y/%m/%d %H:%M:%S',
    format=(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(lineno)d - %(module)s - %(message)s'
    ),
    filemode='w',
)
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


def build_negative_samples(data, item_num):
    """Preserve the original candidate sampling range and order."""
    data['negative_samples'] = None

    for i in range(data['seq'].size):
        candidates = [data['next'][i]]

        while len(candidates) < 101:
            item = random.randint(0, item_num - 1)

            if item not in candidates:
                candidates.append(item)

        data.at[i, 'negative_samples'] = candidates


def main(args):
    fix_random_seed_as(args.random_seed)

    if args.t_dataset in ('toy', 'sports'):
        args.dropout = 0.3

    print(args)
    logger.info(args)

    source_dir = os.path.join('../data', args.s_dataset)
    target_dir = os.path.join('../data', args.t_dataset)

    s_tra_data = pd.read_pickle(os.path.join(source_dir, 'train.df'))
    s_val_data = pd.read_pickle(os.path.join(source_dir, 'val.df'))
    s_test_data = pd.read_pickle(os.path.join(source_dir, 'test.df'))

    t_tra_data = pd.read_pickle(os.path.join(target_dir, 'train.df'))
    t_val_data = pd.read_pickle(os.path.join(target_dir, 'val.df'))
    t_test_data = pd.read_pickle(os.path.join(target_dir, 'test.df'))

    source_item_num = 98507
    target_item_num = 23978
    args = item_num_create(args, source_item_num, target_item_num)

    save_dir = os.path.join(
        './saved_model',
        args.s_dataset + "_" + args.t_dataset,
    )
    os.makedirs(save_dir, exist_ok=True)

    args.source_model_path = os.path.join(
        save_dir, args.model_name + "_source.pth"
    )
    args.target_model_path = os.path.join(
        save_dir, args.model_name + "_target.pth"
    )

    # Keep this order to preserve the original random candidate sequence.
    build_negative_samples(s_val_data, args.source_item_num)
    build_negative_samples(t_val_data, args.target_item_num)
    build_negative_samples(s_test_data, args.source_item_num)
    build_negative_samples(t_test_data, args.target_item_num)

    diffu_rec = create_model_diffu(args)
    rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)

    # Stage 1: Source pretraining
    if args.skip_source_pretrain:
        if args.init_source_model is None:
            raise ValueError(
                '--init_source_model must be provided when '
                '--skip_source_pretrain is enabled.'
            )

        if not os.path.exists(args.init_source_model):
            raise FileNotFoundError(
                'Source checkpoint not found: ' + args.init_source_model
            )

        rec_diffu_joint_model.load_state_dict(
            torch.load(
                args.init_source_model,
                map_location=args.device,
            )
        )

        print('Skipped source pretraining. Loaded:', args.init_source_model)
        logger.info(
            'Skipped source pretraining. Loaded: %s',
            args.init_source_model,
        )

    else:
        args.pretrain_flag = True

        best_model, source_test_results = model_train(
            s_tra_data,
            s_val_data,
            s_test_data,
            t_tra_data,
            rec_diffu_joint_model,
            args,
            logger,
            True,
        )

        if not os.path.exists(args.source_model_path):
            raise FileNotFoundError(
                'Source model was not saved: ' + args.source_model_path
            )

        rec_diffu_joint_model.load_state_dict(
            torch.load(
                args.source_model_path,
                map_location=args.device,
            )
        )

        print('Source model loaded from:', args.source_model_path)
        logger.info(
            'Source model loaded from: %s',
            args.source_model_path,
        )

    # Stage 2: Target fine-tuning / evaluation
    fix_random_seed_as(args.random_seed)

    args.item_num = target_item_num
    args.eval_interval = 1
    args.patience = 5
    args.pretrain_flag = False

    experiment_message = (
        f"Target experiment: "
        f"model_name={args.model_name}, "
        f"tail_augmentation={args.enable_tail_augmentation}, "
        f"tail_replay={args.enable_tail_replay}, "
        f"tail_replay_ratio={args.tail_replay_ratio}, "
        f"tail_replay_strategy={args.tail_replay_strategy}, "
        f"sequence_variant={args.sequence_variant}, "
        f"sequence_aug_prob={args.sequence_aug_prob}, "
        f"preserve_recent={args.sequence_preserve_recent}"
    )
    print(experiment_message)
    logger.info(experiment_message)

    best_model, target_test_results = model_train(
        t_tra_data,
        t_val_data,
        t_test_data,
        None,
        rec_diffu_joint_model,
        args,
        logger,
        False,
    )

    if args.eval_checkpoint is None:
        torch.save(
            best_model.state_dict(),
            args.target_model_path,
        )
        print('Target model saved to:', args.target_model_path)
        logger.info(
            'Target model saved to: %s',
            args.target_model_path,
        )
    else:
        print('Evaluation completed:', args.eval_checkpoint)


if __name__ == '__main__':
    main(args)