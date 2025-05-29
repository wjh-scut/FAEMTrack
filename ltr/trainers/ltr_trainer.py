import os
from collections import OrderedDict
from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
import torch
import torch.nn as nn
import time


def freeze_batchnorm_layers(net):
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, freeze_backbone_bn_layers=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
            freeze_backbone_bn_layers - Set to True to freeze the bach norm statistics in the backbone during training.
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

        self.freeze_backbone_bn_layers = freeze_backbone_bn_layers

        # 记录训练过程中验证数据集的平均loss和最小loss的epoch 2024.12.22
        self.avg_val_loss = 0
        self.min_val_epoch = 0

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""
        # cycle_dataset 方法用于执行训练或验证的数据循环。它遍历数据加载器（loader）中的所有批次数据，对每个批次进行前向传播（forward pass）、计算损失、反向传播（backward pass）以及参数更新（在训练模式下）。此外，该方法还负责统计和打印训练或验证过程中的相关指标。
        self.actor.train(loader.training) #设置训练模式

        if self.freeze_backbone_bn_layers:
            freeze_batchnorm_layers(self.actor.net.feature_extractor)

        torch.set_grad_enabled(loader.training) #设置梯度计算模式（启用梯度计算，用于训练模式；禁用梯度计算，用于验证或测试模式，从而节省内存和计算资源）

        self._init_timing() #初始化与计时相关的变量，以便跟踪每个批次的处理时间或整体训练时间

        for i, data in enumerate(loader, 1): #loader 是数据加载器，包含多个批次的数据。enumerate(loader, 1) 从 1 开始对数据进行编号，i 是当前批次的索引，data 是当前批次的数据。
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update statistics
            self._update_stats(stats, loader.batch_size, loader)

            # print statistics
            self._print_stats(i, loader, loader.batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])
            # ------------------------------------------------ OSTrack保存log的代码, xyl20231215
            log_dir = os.path.join(self.settings.env.tensorboard_dir, 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, "%s-%s-%s-train.log" % (self.settings.script_name, self.settings.module_name, self.settings.idea_name))
            val_log_file = os.path.join(log_dir, "%s-%s-%s-val.log" % (self.settings.script_name, self.settings.module_name, self.settings.idea_name))
            log_str = print_str[:-5] + '\n'
            with open(log_file, 'a') as f:
                f.write(log_str)
            if loader.name == "val":
                with open(val_log_file, 'a') as f:
                    f.write(log_str)
            # ------------------------------------------------ OSTrack保存log的代码, xyl20231215

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
