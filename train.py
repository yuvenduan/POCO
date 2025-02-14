import logging
import os.path as osp
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs

from configs.config_global import LOG_LEVEL, NP_SEED, TCH_SEED, USE_CUDA, DATA_DIR
from configs.configs import BaseConfig, SupervisedLearningBaseConfig, NeuralPredictionConfig
from datasets.dataloader import DatasetIters
from tasks.taskfunctions import TaskFunction
from models.model_utils import model_init
from utils.config_utils import load_config
from utils.logger import Logger
from utils.train_utils import (get_grad_norm, grad_clipping, task_init, log_complete)
from datetime import datetime

def train_from_path(path):
    """Train from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_train(config)

def eval_from_path(path):
    """Evaluate from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_eval(config)

def evaluate_performance(
    net: nn.Module, 
    config: SupervisedLearningBaseConfig, 
    test_data: DatasetIters, 
    task_func: TaskFunction, 
    logger: Logger, 
    testloss_list: list = [], 
    i_b: int = 0,
    train_loss: float = 0,
    phase: str = 'val',
    testing: bool = False
):
    """
    Test the model. Print the test loss and accuracy through logger. Save the model if it is the best model.
    """

    net.eval()
    with torch.no_grad():

        test_data.reset()
        all_loss = 0

        for i_tloader, test_iter in enumerate(test_data.data_iters):
            total = 0
            test_loss = 0.0
            while True:
                try:
                    t_data = next(test_iter)
                    t_data = get_full_input(t_data, i_tloader, config, test_data.input_sizes)
                    result = task_func.roll(net, t_data, phase)
                    loss, num = result[: 2]
                    test_loss += loss.item() * num
                    total += num
                except StopIteration:
                    break
            dataset_loss = test_loss / total
            if config.log_loss:
                dataset_loss = np.log(dataset_loss + 1e-4)
            all_loss += dataset_loss * config.mod_w[i_tloader]
        avg_testloss = all_loss / sum(config.mod_w)
    
    logger.log_tabular('TestLoss', avg_testloss)
    testloss_list.append(avg_testloss)

    # save the model with best testing loss
    if not testing:
        best = False
        if avg_testloss <= min(testloss_list):
            best = True
            torch.save(net.state_dict(), osp.join(config.save_path, 'net_best.pth'))
    else:
        best = True

    task_func.after_testing_callback(
        logger=logger, save_path=config.save_path, is_best=best, batch_num=i_b, testing=testing
    )

def get_full_input(batch, dataset_idx, config: NeuralPredictionConfig, input_sizes):
    """
    Use the input list for one dataset to create the full input list for all datasets.
    """
    # prepare an empty list
    input_size_list = list(itertools.chain(*input_sizes))
    n_sessions = [len(input_size) for input_size in input_sizes]
    dataset_start_idx = [sum(n_sessions[:i]) for i in range(len(n_sessions) + 1)]

    batch_input, batch_target, batch_info = batch

    input_list = []
    target_list = []
    info_list = []

    for i, size in enumerate(input_size_list):
        if i >= dataset_start_idx[dataset_idx] and i < dataset_start_idx[dataset_idx + 1]:
            idx = i - dataset_start_idx[dataset_idx]
            input_list.append(batch_input[idx])
            target_list.append(batch_target[idx])
            info_list.append(batch_info[idx])
        else:
            input_list.append(torch.zeros(config.seq_length - config.pred_length, 0, size))
            target_list.append(torch.zeros(config.seq_length - 1, 0, size))
            info_list.append({'session_idx': i, })

    return input_list, target_list, info_list

def model_train(config: NeuralPredictionConfig):
    """
    The main training function. 
    This function initializes the task, dataset, network, optimizer, and learning rate scheduler.
    It then trains the network and logs the performance.
    """

    np.random.seed(NP_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    random.seed(config.seed)
    # set the torch hub directory
    torch.hub.set_dir(osp.join(DATA_DIR, 'torch_hub'))
    start_time = datetime.now()

    assert config.config_mode == 'train', 'config mode must be train'
    if USE_CUDA:
        logging.info("training with GPU")

    # initialize logger
    logger = Logger(output_dir=config.save_path,
                    exp_name=config.experiment_name)

    # initialize dataset
    train_data = DatasetIters(config, 'train')
    assert config.perform_val
    test_data = DatasetIters(config, 'val')
    test_baseline_performance = test_data.get_baselines()

    # initialize task
    task_func: TaskFunction = task_init(config, train_data.input_sizes)
    task_func.mse_baseline['val'] = test_baseline_performance

    # initialize model
    net = model_init(config, train_data.input_sizes, train_data.unit_types)
    
    # initialize optimizer
    if config.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.wdecay)
    elif config.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.wdecay, amsgrad=True)
    elif config.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr_SGD,
                                    momentum=0.9, weight_decay=config.wdecay)
    else:
        raise NotImplementedError('optimizer not implemented')

    # initialize Learning rate scheduler
    if config.use_lr_scheduler:
        if config.scheduler_type == 'ExponentialLR':
            scheduler = lrs.ExponentialLR(optimizer, gamma=0.1)
        elif config.scheduler_type == 'StepLR':
            scheduler = lrs.StepLR(optimizer, 1.0, gamma=0.1)
        elif config.scheduler_type == 'CosineAnnealing':
            scheduler = lrs.CosineAnnealingLR(
                optimizer, 
                T_max=config.max_batch // config.log_every + 1,
                eta_min=config.lr / 20
            )
        else:
            raise NotImplementedError('scheduler_type must be specified')

    i_b = 0
    i_log = 0
    testloss_list = []
    break_flag = False
    train_loss = 0.0

    for epoch in range(config.num_ep):
        train_data.reset()

        for step_ in range(train_data.min_iter_len):
            
            net.train()
            # save model
            if (i_b + 1) % config.save_every == 0:
                torch.save(net.state_dict(), osp.join(config.save_path, 'net_{}.pth'.format(i_b + 1)))

            loss = 0.0
            optimizer.zero_grad()

            for i_loader, train_iter in enumerate(train_data.data_iters):
                mod_weight = config.mod_w[i_loader]
                data = next(train_iter)
                data = get_full_input(data, i_loader, config, train_data.input_sizes)
                dataset_loss = task_func.roll(net, data, 'train')
                if config.log_loss:
                    dataset_loss = torch.log(dataset_loss + 1e-4)
                loss += dataset_loss * mod_weight

            loss.backward()
            # gradient clipping
            if config.grad_clip is not None:
                grad_clipping(net, config.grad_clip)

            optimizer.step()
            train_loss += loss.item()            

            # log performance
            if i_b % config.log_every == config.log_every - 1:

                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('BatchNum', i_b + 1)
                logger.log_tabular('DataNum', (i_b + 1) * config.batch_size * train_data.num_datasets)
                logger.log_tabular('TrainLoss', train_loss / config.log_every)

                evaluate_performance(net, config, test_data, task_func, logger, testloss_list, i_b + 1, train_loss / config.log_every)
                i_log += 1
                logger.dump_tabular()
                train_loss = 0

                if config.use_lr_scheduler:
                    scheduler.step()

            i_b += 1
            if i_b >= config.max_batch:
                break_flag = True
                break

        if break_flag:
            break

    task_func.after_training_callback(config, net)
    log_complete(config.save_path, start_time)

    if config.perform_test:
        model_eval(config)

def model_eval(config: SupervisedLearningBaseConfig):
    np.random.seed(NP_SEED)
    torch.manual_seed(TCH_SEED)
    random.seed(config.seed)

    test_data = DatasetIters(config, 'test')
    task_func: TaskFunction = task_init(config, test_data.input_sizes)
    task_func.mse_baseline['val'] = test_data.get_baselines()
    net = model_init(config, test_data.input_sizes, test_data.unit_types)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), weights_only=True))

    logger = Logger(output_dir=config.save_path, output_fname='test.txt', exp_name=config.experiment_name)
    evaluate_performance(net, config, test_data, task_func, logger, testing=True)
    logger.dump_tabular()