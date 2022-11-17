# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/06/23
    Description:
        将Network包装为trainOneStep
"""
import os
import sys

current_path = os.path.dirname(__file__)
sys.path.append(current_path)
# print("listdir", os.listdir(current_path))
os.system(f'bash {current_path}/install_third_part.sh')

import datetime, os, argparse, pickle, shutil
from pathlib import Path
import numpy as np

import mindspore as ms
from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net, nn, ops, set_seed
from mindspore.nn import Adam
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig, Callback
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore import dtype as mstype
from mindspore.profiler import Profiler
# import mindspore.nn as nn
# nn.TrainOneStepCell

from data.S3DIS_dataset import dataloader, ms_map
from model.base_model import get_param_groups
# from model.model_s3dis import RandLANet_S3DIS, RandLA_S3DIS_WithLoss
# from
from utils.tools import DataProcessing as DP
from utils.tools import ConfigS3DIS as cfg
from utils.logger import get_logger

import time
from time import strftime, localtime

use_custom_train_one_step_cell = True  # 如果为True,则使用CustomTrainOneStepCell


class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer, sens=1.0):
        """入参有三个：训练网络，优化器和反向传播缩放比例"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network  # 定义前向网络
        self.network.set_grad()  # 构建反向网络
        self.optimizer = optimizer  # 定义优化器
        self.weights = self.optimizer.parameters  # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        # 执行前向网络，计算当前输入的损失函数值
        # begin_time = time.time()
        loss = self.network(*inputs)
        # forward_time = time.time()
        # print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
        #       f"[train_modelarts_notebook_v2.py] forward 耗时: {forward_time - begin_time}s")
        # 进行反向传播，计算梯度
        grads = self.grad(self.network, self.weights)(*inputs, loss)

        # backward_time = time.time()
        # print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
        #       f"[train_modelarts_notebook_v2.py] backward 耗时: {backward_time - forward_time}s")
        # 使用优化器更新梯度
        loss = ops.depend(loss, self.optimizer(grads))
        # print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
        #       f"[train_modelarts_notebook_v2.py] update optimizer 耗时: {time.time() - backward_time}s")
        # print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
        #       f"[train_modelarts_notebook_v2.py] forward+backward+update optimizer 耗时: {time.time() - begin_time}s")
        return loss


class UpdateLossEpoch(Callback):
    def __init__(self, num_training_ep0=30, logger=None):
        super(UpdateLossEpoch, self).__init__()
        self.training_ep = {i: np.exp(i / 100 - 1.0) - np.exp(-1.0) for i in range(0, 100)}
        self.training_ep.update({i: 0 for i in range(0, num_training_ep0)})
        self.logger = logger

    # v1.8: on_train_epoch_begin
    # v1.7: epoch_begin
    def epoch_begin(self, run_context):
        # update_loss_time = time.time()

        cb_params = run_context.original_args()
        if use_custom_train_one_step_cell:
            train_network_with_loss = cb_params.network.network
        else:
            train_network_with_loss = cb_params.network

        cur_epoch_num = cb_params.cur_epoch_num  # 从1开始
        train_network_with_loss.c_epoch_k += self.training_ep[cur_epoch_num - 1]

        self.logger.info(
            f"UpdateLossEpoch ==>  cur_epoch_num:{cur_epoch_num}, "
            f"cur_training_ep:{self.training_ep[cur_epoch_num]}, "
            f"loss_fn.c_epoch_k:{train_network_with_loss.c_epoch_k}")

        # print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
        #       f"[train_gpu.py] UpdateLossEpoch 耗时: {time.time() - update_loss_time}s")


class S3DISLossMonitor(Callback):
    def __init__(self, per_print_times=1, logger=None):
        super(S3DISLossMonitor, self).__init__()
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.logger = logger

    # v1.8: on_train_step_end
    # v1.7: step_end
    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(f"epoch: {cb_params.cur_epoch_num} "
                             f"step: {cur_step_in_epoch}. "
                             f"Invalid loss {loss}, terminating training.")

        # In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -= \
                    max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            # self.train_network_with_loss = cb_params.network

            msg = f"epoch: {cb_params.cur_epoch_num} " \
                  f"step: {cur_step_in_epoch}, " \
                  f"loss is {loss} "
            self.logger.info(msg)


def prepare_network(weights, cfg, args):
    """Prepare Network"""
    # from model.model_s3dis import RandLANet_S3DIS, RandLA_S3DIS_WithLoss
    from model.model_s3dis_remove_bias import RandLANet_S3DIS, RandLA_S3DIS_WithLoss
    d_in = 6  # xyzrgb
    network = RandLANet_S3DIS(d_in, cfg.num_classes)
    if args.ss_pretrain:
        print(f"Load scannet pretrained ckpt from {args.ss_pretrain}")
        param_dict = load_checkpoint(args.ss_pretrain)
        whitelist = ["encoder"]
        load_all = True
        new_param_dict = dict()
        for key, val in param_dict.items():
            if key.split(".")[0] == 'network' and key.split(".")[1] in whitelist:
                new_key = ".".join(key.split(".")[1:])
                new_param_dict[new_key] = val
        load_param_into_net(network, new_param_dict, strict_load=True)

    network = RandLA_S3DIS_WithLoss(network, weights, cfg.num_classes, cfg.ignored_label_indexs, cfg.c_epoch,
                                    cfg.loss3_type, cfg.topk)

    if args.retrain_model:
        print(f"Load S3DIS pretrained ckpt from {args.retrain_model}")
        param_dict = load_checkpoint(args.retrain_model)
        load_param_into_net(network, param_dict, strict_load=True)

    return network


def prepare_network_graph(weights, cfg, args):
    """Prepare Network"""
    from model.model_s3dis_graph import RandLANet_S3DIS, RandLA_S3DIS_WithLoss
    d_in = 6  # xyzrgb
    network = RandLANet_S3DIS(d_in, cfg.num_classes)
    if args.ss_pretrain:
        print(f"Load scannet pretrained ckpt from {args.ss_pretrain}")
        param_dict = load_checkpoint(args.ss_pretrain)
        whitelist = ["encoder"]
        load_all = True
        new_param_dict = dict()
        for key, val in param_dict.items():
            if key.split(".")[0] == 'network' and key.split(".")[1] in whitelist:
                new_key = ".".join(key.split(".")[1:])
                new_param_dict[new_key] = val
        load_param_into_net(network, new_param_dict, strict_load=True)

    network = RandLA_S3DIS_WithLoss(network, weights, cfg.num_classes, cfg.ignored_label_indexs, cfg.c_epoch,
                                    cfg.loss3_type, cfg.topk)

    if args.retrain_model:
        print(f"Load S3DIS pretrained ckpt from {args.retrain_model}")
        param_dict = load_checkpoint(args.retrain_model)
        load_param_into_net(network, param_dict, strict_load=True)

    return network


def train(cfg, args):
    if cfg.graph_mode:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)

    profiler = Profiler(output_path='./profiler_data')
    ##
    logger = get_logger(args.outputs_dir, args.rank)

    logger.info("============ Args =================")
    for arg in vars(args):
        logger.info('%s: %s' % (arg, getattr(args, arg)))
    logger.info("============ Cfg =================")
    for c in vars(cfg):
        logger.info('%s: %s' % (c, getattr(cfg, c)))

    train_loader, val_loader, dataset = dataloader(cfg, shuffle=False, num_parallel_workers=8)
    ignored_label_indexs = [getattr(dataset, 'label_to_idx')[ign_label] for ign_label in
                            getattr(dataset, 'ignored_labels')]
    cfg.ignored_label_indexs = ignored_label_indexs
    weights = DP.get_class_weights("S3DIS")

    if cfg.graph_mode:
        network = prepare_network_graph(weights, cfg, args)
    else:
        network = prepare_network(weights, cfg, args)
    decay_lr = nn.ExponentialDecayLR(cfg.learning_rate, cfg.lr_decays, decay_steps=cfg.train_steps, is_stair=True)
    opt = Adam(
        params=get_param_groups(network),
        learning_rate=decay_lr,
        loss_scale=cfg.loss_scale
    )

    log = {'cur_epoch': 1, 'cur_step': 1, 'best_epoch': 1, 'besr_miou': 0.0}
    if not os.path.exists(args.outputs_dir + '/log.pkl'):
        f = open(args.outputs_dir + '/log.pkl', 'wb')
        pickle.dump(log, f)
        f.close()

    # resume checkpoint, cur_epoch, best_epoch, cur_step, best_step
    if args.resume:
        f = open(args.resume + '/log.pkl', 'rb')
        log = pickle.load(f)
        print(f"log of resume file {log}")
        f.close()
        param_dict = load_checkpoint(args.resume)
        load_param_into_net(network, param_dict)
        # load_param_into_net(network, args.resume)

    # data loader

    train_loader = train_loader.batch(batch_size=cfg.batch_size,
                                      per_batch_map=ms_map,
                                      input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
                                      output_columns=["features", "features2", "labels", "input_inds", "cloud_inds",
                                                      "p0", "p1", "p2", "p3", "p4",
                                                      "n0", "n1", "n2", "n3", "n4",
                                                      "pl0", "pl1", "pl2", "pl3", "pl4",
                                                      "u0", "u1", "u2", "u3", "u4",
                                                      # 'test_dict'
                                                      ],
                                      drop_remainder=True)

    logger.info('==========begin training===============')

    # loss scale manager
    loss_scale = cfg.loss_scale
    # loss_scale = args.scale_weight
    loss_scale_manager = FixedLossScaleManager(loss_scale) if args.scale or loss_scale != 1.0 else None
    print('loss_scale:', loss_scale)

    # float 16
    if cfg.float16:
        print("network uses float16")
        network.to_float(mstype.float16)

    if args.scale:
        model = Model(network,
                      loss_scale_manager=loss_scale_manager,
                      loss_fn=None,
                      optimizer=opt)
    else:
        if use_custom_train_one_step_cell:
            network = CustomTrainOneStepCell(network, opt)
            model = Model(network)
        else:
            model = Model(network,
                          loss_fn=None,
                          optimizer=opt,
                          )

    # callback for loss & time cost
    loss_cb = S3DISLossMonitor(5, logger)
    time_cb = TimeMonitor(data_size=cfg.train_steps)
    cbs = [loss_cb, time_cb]

    # callback for saving ckpt
    config_ckpt = CheckpointConfig(save_checkpoint_steps=cfg.train_steps, keep_checkpoint_max=100)
    ckpt_cb = ModelCheckpoint(prefix='randla',
                              directory=os.path.join(args.outputs_dir, 'ckpt'),
                              config=config_ckpt)
    cbs += [ckpt_cb]

    update_loss_epoch_cb = UpdateLossEpoch(args.num_training_ep0, logger)
    cbs += [update_loss_epoch_cb]

    # profiler callback
    # profiler_cb = ProfileStopAtStep()
    # cbs += [profiler_cb]

    logger.info(f"Outputs_dir:{args.outputs_dir}")
    logger.info(f"Total number of epoch: {cfg.max_epoch}; "
                f"Dataset capacity: {train_loader.get_dataset_size()}")

    model.train(cfg.max_epoch,
                train_loader,
                callbacks=cbs,  # [loss_cb,time_cb,ckpt_cb,update_loss_epoch_cb]
                dataset_sink_mode=False)
    profiler.analyse()
    logger.info('==========end training===============')


if __name__ == "__main__":
    """Parse program arguments"""
    parser = argparse.ArgumentParser(
        prog='RandLA-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    expr.add_argument('--epochs', type=int, help='max epochs', default=100)

    expr.add_argument('--batch_size', type=int, help='batch size', default=6)

    expr.add_argument('--dataset_dir', type=str, help='path of dataset', default='./datasets/S3DIS')

    expr.add_argument('--outputs_dir', type=str, help='path of output', default='outputs')

    expr.add_argument('--val_area', type=str, help='area to validate', default='Area_5')

    expr.add_argument('--resume', type=str, help='model to resume', default=None)

    expr.add_argument('--scale', type=bool, help='scale or not', default=False)

    # expr.add_argument('--scale_weight', type=float, help='scale weight', default=1.0)

    misc.add_argument('--device_target', type=str, help='CPU | GPU | Ascend ', default='Ascend')

    misc.add_argument('--device_id', type=int, help='GPU id to use', default=0)

    misc.add_argument('--rank', type=int, help='rank', default=0)

    misc.add_argument('--name', type=str, help='name of the experiment',
                      default=None)
    misc.add_argument('--ss_pretrain', type=str, help='name of the experiment',
                      default=None)
    misc.add_argument('--retrain_model', type=str, help='name of the experiment',
                      default=None)
    misc.add_argument('--float16', type=bool, default=False)

    misc.add_argument('--train_steps', type=int, default=500)
    misc.add_argument('--learning_rate', type=float, default=0.01)
    misc.add_argument('--lr_decays', type=float, default=0.95)
    misc.add_argument('--loss_scale', type=float, default=1.0)
    misc.add_argument('--topk', type=int, default=500)
    misc.add_argument('--num_training_ep0', type=int, default=30)
    misc.add_argument('--labeled_percent', type=int, default=1)  # range in [1,100]
    misc.add_argument('--random_seed', type=int, default=888)
    misc.add_argument('--graph_mode', action='store_true', default=False)

    args = parser.parse_args()

    cfg.dataset_dir = args.dataset_dir
    cfg.batch_size = args.batch_size
    cfg.max_epoch = args.epochs
    cfg.train_steps = args.train_steps
    cfg.learning_rate = args.learning_rate
    cfg.lr_decays = args.lr_decays
    cfg.loss_scale = args.loss_scale
    cfg.topk = args.topk
    num_training_ep0 = args.num_training_ep0
    cfg.training_ep0 = {i: 0 for i in range(0, num_training_ep0)}
    cfg.training_ep = {i: np.exp(i / 100 - 1.0) - np.exp(-1.0) for i in range(0, 100)}
    cfg.training_ep.update(cfg.training_ep0)
    cfg.labeled_percent = args.labeled_percent
    cfg.random_seed = args.random_seed
    cfg.graph_mode = args.graph_mode
    cfg.float16 = args.float16

    if args.name is None:
        if args.resume:
            args.name = Path(args.resume).split('/')[-1]
        else:
            time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
            args.name = f'TSteps{cfg.train_steps}_MaxEpoch{cfg.max_epoch}_BatchS{cfg.batch_size}_lr{cfg.learning_rate}' \
                        f'_lrd{cfg.lr_decays}_ls{cfg.loss_scale}_Topk{cfg.topk}_NumTrainEp0{num_training_ep0}_LP_{cfg.labeled_percent}_RS_{cfg.random_seed}'
            if cfg.graph_mode:
                args.name += "_GraphM"
            else:
                args.name += "_PyNateiveM"
            args.name += f'_{time_str}'

    np.random.seed(cfg.random_seed)
    set_seed(cfg.random_seed)
    # https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.set_seed.html?highlight=set_seed

    # ds.config.set_auto_num_workers()
    # output_dir = f"./runs/pretrain_s3dis_v13"
    args.outputs_dir = os.path.join(args.outputs_dir, args.name)

    print(f"outputs_dir:{args.outputs_dir}")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    if args.resume:
        args.outputs_dir = args.resume

    # copy file
    # shutil.copy('utils/tools.py', str(args.outputs_dir))
    # shutil.copy('train_gpu.py', str(args.outputs_dir))
    # shutil.copy('model/model_s3dis.py', str(args.outputs_dir))
    # shutil.copy('data/S3DIS_dataset_v1.py', str(args.outputs_dir))
    #

    # start train

    train(cfg, args)
