import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn
import random
import numpy as np


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import ltr.admin.settings as ws_settings

# 2025.03.30
def set_seed(seed=3407):
    """
    固定随机种子，确保实验结果可复现

    参数:
    - seed (int): 要设置的随机种子，默认值为 3407
    """
    # 固定 Python 内置的随机模块
    random.seed(seed)

    # 固定 NumPy 的随机种子
    np.random.seed(seed)

    # 固定 PyTorch 的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 设置 CUDA 后端为确定性模式（可能会牺牲部分性能）
    torch.backends.cudnn.deterministic = True #可能会让训练变得非常慢，可以考虑删除
    torch.backends.cudnn.benchmark = False

    # 固定 Python 哈希种子（确保内置数据结构的哈希结果一致）
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_training(train_module, train_name, idea_name, cudnn_benchmark=True):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.idea_name = idea_name
    settings.project_path = '{}/{}/{}'.format(idea_name, train_module, train_name)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')  # 获取train_settings中py文件的run函数

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--train_module', type=str, default='dimp', help='Name of module in the "train_settings/" folder.')
    parser.add_argument('--train_name', type=str, default='super_dimp', help='Name of the train settings file.')
    parser.add_argument('--idea_name', type=str, default='FAEMTrack', help='Name of the idea.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    # 添加一个用于设置随机种子的参数 2025.03.30
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # 在调用任何可能产生随机性的操作之前，先固定随机种子 2025.03.30
    set_seed(args.seed)

    run_training(args.train_module, args.train_name, args.idea_name, args.cudnn_benchmark)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
