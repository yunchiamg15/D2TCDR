"""Five methods, one frozen Target-domain experiment protocol."""
import argparse
from pathlib import Path

import pandas as pd
import torch

from model import Att_Diffuse_model, create_model_diffu
from protocol import digest_file, prepare_protocol, seed_all
from trainer import train_method

METHOD_NAMES = dict(baseline='Shared_Baseline',random_b3='Shared_RandomB3',
    v3_selection='Shared_V3Selection',copy_replace='Shared_CopyReplace',four_head='Shared_FourHead')


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',required=True,choices=list(METHOD_NAMES))
    parser.add_argument('--model_name',default=None)
    parser.add_argument('--data_root',default='../data')
    parser.add_argument('--s_dataset',default='movie')
    parser.add_argument('--t_dataset',default='video')
    parser.add_argument('--init_source_model',default='/mnt/mira/D2TCDR/src/saved_model/movie_video/traditional_source.pth')
    parser.add_argument('--skip_source_pretrain',action='store_true',help='Compatibility flag: all shared experiments load a fixed Source checkpoint')
    parser.add_argument('--output_dir',default=None)
    parser.add_argument('--protocol_file',default=None)
    parser.add_argument('--reward_reference',default=None)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--num_gpu',type=int,default=1)
    parser.add_argument('--source_item_num',type=int,default=98507)
    parser.add_argument('--target_item_num',type=int,default=23978)
    parser.add_argument('--random_seed',type=int,default=1997,help='Fixed split and epoch shuffle')
    parser.add_argument('--target_seed',type=int,default=1998,help='Recommendation model training RNG, shared by all methods')
    parser.add_argument('--eval_seed',type=int,default=2026)
    parser.add_argument('--augmentation_seed',type=int,default=3997)
    parser.add_argument('--reward_holdout_ratio',type=float,default=.1)
    parser.add_argument('--reward_eval_samples',type=int,default=0,choices=[0],help='Use all fixed RES samples')
    parser.add_argument('--popular_ratio',type=float,default=.2)
    parser.add_argument('--max_len',type=int,default=8)
    parser.add_argument('--batch_size',type=int,default=512)
    parser.add_argument('--hidden_size',type=int,default=128)
    parser.add_argument('--num_blocks',type=int,default=4)
    parser.add_argument('--hidden_act',default='gelu')
    parser.add_argument('--dropout',type=float,default=.1)
    parser.add_argument('--emb_dropout',type=float,default=.3)
    parser.add_argument('--p',type=float,default=.3)
    parser.add_argument('--w',type=float,default=1.)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--optimizer',choices=['Adam'],default='Adam')
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--weight_decay',type=float,default=0.)
    parser.add_argument('--loss_lambda',type=float,default=1.)
    parser.add_argument('--decay_step',type=int,default=100)
    parser.add_argument('--gamma',type=float,default=.1)
    parser.add_argument('--eval_interval',type=int,default=1)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--metric_ks',nargs='+',type=int,default=[5,10,20])
    parser.add_argument('--schedule_sampler_name',default='lossaware')
    parser.add_argument('--diffusion_steps',type=int,default=32)
    parser.add_argument('--lambda_uncertainty',type=float,default=.001)
    parser.add_argument('--noise_schedule',default='trunc_lin')
    parser.add_argument('--rescale_timesteps',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--preserve_recent',type=int,default=2)
    parser.add_argument('--replacement_topk',type=int,default=10)
    parser.add_argument('--min_cooccurrence',type=int,default=2)
    parser.add_argument('--agent_budget_ratio',type=float,default=.2)
    parser.add_argument('--agent_lr',type=float,default=.0001)
    parser.add_argument('--agent_hidden',type=int,default=32)
    parser.add_argument('--agent_entropy_coef',type=float,default=0.,help='0 preserves original V3 REINFORCE')
    parser.add_argument('--reward_baseline_window',type=int,default=50)
    parser.add_argument('--reward_lambda_overall',type=float,default=1.)
    parser.add_argument('--reward_lambda_popular',type=float,default=1.)
    parser.add_argument('--reward_epsilon_overall',type=float,default=.002)
    parser.add_argument('--reward_epsilon_popular',type=float,default=.002)
    args=parser.parse_args()
    if args.num_gpu!=1:
        parser.error('Shared prototype supports one device')
    if not 0<args.reward_holdout_ratio<1 or not 0<args.popular_ratio<1 or not 0<=args.agent_budget_ratio<=1:
        parser.error('Invalid split/popularity/budget ratio')
    if min(args.batch_size,args.max_len,args.epochs,args.eval_interval,args.patience,args.decay_step,args.agent_hidden,args.reward_baseline_window,args.replacement_topk,args.min_cooccurrence)<1 or args.preserve_recent<2:
        parser.error('Invalid positive size/interval')
    if args.lr<=0 or args.agent_lr<=0 or args.agent_entropy_coef<0:
        parser.error('Invalid learning rate/entropy coefficient')
    if min(args.reward_lambda_overall,args.reward_lambda_popular,args.reward_epsilon_overall,args.reward_epsilon_popular)<0:
        parser.error('Reward penalties/tolerances must be nonnegative')
    if 10 not in args.metric_ks or min(args.metric_ks)<1 or max(args.metric_ks)>101:
        parser.error('metric_ks must contain 10 and be within 1..101')
    args.metric_ks=sorted(set(args.metric_ks))
    args.model_name=args.model_name or METHOD_NAMES[args.method]
    if Path(args.model_name).name!=args.model_name or args.model_name in ('','.','..'):
        parser.error('model_name must be a filename, not a path')
    args.output_dir=args.output_dir or f'./saved_model/{args.s_dataset}_{args.t_dataset}/shared_v1'
    args.protocol_file=args.protocol_file or str(Path(args.output_dir)/'shared_protocol_v1.json')
    args.reward_reference=args.reward_reference or str(Path(args.output_dir)/'Shared_Baseline_target_reward_reference.json')
    return args


def main(args):
    source_path=Path(args.init_source_model)
    if not source_path.is_file():
        raise FileNotFoundError(f'Source checkpoint missing: {source_path}')
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu for small checks')
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    seed_all(args.random_seed)
    folder=Path(args.data_root)/args.t_dataset
    frames=[pd.read_pickle(folder/name) for name in ('train.df','val.df','test.df')]
    train,val,res,test,popular,tail,protocol=prepare_protocol(*frames,args,digest_file(source_path))
    print('Method:',args.method)
    print('Shared protocol fingerprint:',protocol['fingerprint'])
    print('Data sizes:',dict(main=len(train),res=len(res),validation=len(val),test=len(test)))
    print('Fixed popularity:',dict(popular=len(popular),tail=len(tail),observed=len(popular|tail)))
    # Construct the original recommendation architecture and load identical Source weights.
    args.item_num=args.source_item_num
    args.pretrain_flag=True
    model=Att_Diffuse_model(create_model_diffu(args),args)
    state=torch.load(source_path,map_location='cpu',weights_only=True)
    model.load_state_dict(state)
    args.item_num=args.target_item_num
    args.pretrain_flag=False
    model=model.to(args.device)
    return train_method(model,train,val,res,test,popular,tail,protocol,args)


if __name__=='__main__':
    main(parse_args())
