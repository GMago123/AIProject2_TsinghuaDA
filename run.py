import os
import argparse
import pandas as pd
import numpy as np
import re
import random
import torch
from tqdm import tqdm
import datetime
from model import Model
from eval import eval
from config import Config
from train import Train


def setup_seed(seed):
    """ Manually Fix the random seed to get deterministic results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str, default='unet',
                        help='support unet')
    parser.add_argument('--loss', type=str, default='bceloss',
                        help='support bceloss, weighted bceloss')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='path to pretrained parameters.')
    parser.add_argument('--model_save_path', type=str, default='results/model_saves',
                        help='path to save model.')
    parser.add_argument('--eval', type=str, default=None,
                        help='path to model parameters.')
    parser.add_argument('--res_save_path', type=str, default='results/result_saves',
                        help='path to save results.')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='path to data directory.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used.')
    return parser.parse_args()


def run(args):
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    args.save = True

    net = Model(args.modelName, args)

    if args.pretrained_path:
        net.load_state_dict(torch.load(args.pretrained_path))
    net.to(args.device)
    
    # 使用并行gpu
    if args.using_cuda and len(args.gpu_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=args.gpu_ids, output_device=args.gpu_ids[0])
    
    # 模型训练
    train_ = Train(net, args)
    train_.train_net()
    
    # 加载最近一次训练的模型进行训练集/测试集上指标的计算
    valid_paths = []
    for path in os.listdir(args.model_save_path):
        affix = f'{args.modelName}-' + datetime.datetime.now().strftime('%Y-%m-%d') + '-'
        matchObj = re.match(affix + r'epoch(\d+).pth', path)
        if matchObj:
            valid_paths.append((int(matchObj.group(1)), matchObj.group())) # (epoch号,文件名)
    valid_paths = sorted(valid_paths)
    load_model_path = os.path.join(args.model_save_path, valid_paths[-1][1])  # 序号最大的model

    net.load_state_dict(torch.load(load_model_path))
    net.to(args.device)
    results = train_.result(net=net, data_dir=args.data_dir, args=args)  # 暂定计算训练集上的指标
    return results

def run_trains(seeds, args):
    model_results = []
    
    for i, seed in enumerate(seeds):
        setup_seed(seed)

        args.cur_time = i + 1
        args.seed = seed   
        print(args)
        test_results = run(args)
        model_results.append(test_results)

    criterions = list(model_results[0].keys())    # 计算出的各项指标
    configs = list(args.keys())     # 配置
    df = pd.DataFrame(columns=["Model"] + criterions + configs)

    res = [args.modelName]
    for c in criterions:
        values = [result[c] for result in model_results]
        mean = round(np.mean(values), 2)
        std = round(np.std(values), 2)
        res.append((mean, std))
    for option in configs:
        res.append(args[option])
    df.loc[len(df)] = res

    save_path = os.path.join(args.res_save_path, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')\
                 + '-' + args.modelName + '.csv')
    if not os.path.exists(args.res_save_path):
        os.makedirs(args.res_save_path)
    df.to_csv(save_path, index=None)
    print(f'Results are saved to {save_path}...')
    

def recognize(seeds=[]):       # 配置好GPU，处理args，识别程序运行模式
    args = parse_args()
    config = Config(args)
    args = config.get_config()

    args.using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if args.using_cuda else 'cpu')
    args.device = device

    if args.eval:
        eval(args)
    else:
        run_trains(seeds, args)


if __name__ == "__main__":
    recognize(seeds=[1111])
    # recognize(seeds=[11, 111, 1111])
