import logging
import os.path as osp
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs

from configs.config_global import LOG_LEVEL, NP_SEED, TCH_SEED, USE_CUDA, DATA_DIR
from configs.configs import BaseConfig, SupervisedLearningBaseConfig
from datasets.data_sets import DatasetIters
from tasks.taskfunctions import TaskFunction
from models.model_utils import model_init
from utils.config_utils import load_config
from utils.logger import Logger
from utils.train_utils import (get_grad_norm, grad_clipping, task_init, log_complete)

def train_from_path(path):
    """Train from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_train(config)

def model_test(
    net: nn.Module, 
    config: SupervisedLearningBaseConfig, 
    test_data: DatasetIters, 
    task_func: TaskFunction, 
    logger: Logger, 
    testloss_list: list = [], 
    i_b: int = 0,
    train_loss: float = 0,
    phase: str = 'val'
):
    """
    Test the model. Print the test loss and accuracy through logger. Save the model if it is the best model.
    """

    extra_out = []
    net.eval()
                
    if config.perform_val:
        correct = 0
        total = 0
        test_loss = 0.0
        test_b = 0

        with torch.no_grad():
            test_data.reset()
            
            for t_step_ in range(test_data.min_iter_len):
                loss_weighted, num_weighted, num_corr_weighted = 0, 0, 0
                for i_tloader, test_iter in enumerate(test_data.data_iters):
                    mod_weight = config.mod_w[i_tloader]
                    # net.set_mode(i_tloader)

                    t_data = next(test_iter)

                    result = task_func.roll(net, t_data, phase)
                    loss, num, num_corr = result[: 3]
                    extra_out.append(result)

                    loss *= mod_weight
                    num *= mod_weight
                    num_corr *= mod_weight

                    loss_weighted += loss
                    num_weighted += num
                    num_corr_weighted += num_corr

                test_loss += loss_weighted.item()
                total += num_weighted
                correct += num_corr_weighted

                test_b += 1
                if test_b >= config.test_batch:
                    break

        if config.print_mode == 'accuracy':
            test_acc = 100 * correct / total
        else:
            test_error = correct / total
        avg_testloss = test_loss / test_b

    else:
        avg_testloss = train_loss
    
    logger.log_tabular('TestLoss', avg_testloss)
    testloss_list.append(avg_testloss)
    # save the model with best testing loss

    best = False
    if avg_testloss <= min(testloss_list):
        best = True
        torch.save(net.state_dict(),
                    osp.join(config.save_path, 'net_best.pth'))

    if config.perform_val:
        if config.print_mode == 'accuracy':
            logger.log_tabular('TestAcc', test_acc)
        else:
            logger.log_tabular('TestError', test_error)

        task_func.after_testing_callback(
            batch_info=extra_out, logger=logger, 
            save_path=config.save_path, is_best=best, batch_num=i_b
        )

def model_train(config: SupervisedLearningBaseConfig):
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

    # gradient clipping
    if config.grad_clip is not None:
        logging.info("Performs grad clipping with max norm " + str(config.grad_clip))

    # initialize dataset
    train_data = DatasetIters(config, 'train')
    if config.perform_val:
        test_data = DatasetIters(config, 'val')
    else:
        test_data = None

    # initialize task
    task_func: TaskFunction = task_init(config, train_data.datum_sizes[0])

    # initialize network
    net = model_init(config, train_data.datum_sizes[0])

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
                eta_min=config.lr / 10
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
                # net.set_mode(i_loader)

                data = next(train_iter)
                loss += task_func.roll(net, data, 'train') * mod_weight

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

                model_test(net, config, test_data, task_func, logger, testloss_list, i_b + 1, train_loss / config.log_every)
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
    if config.perform_test:
        np.random.seed(NP_SEED)
        torch.manual_seed(TCH_SEED)
        random.seed(config.seed)

        test_data = DatasetIters(config, 'test')
        logger = Logger(output_dir=config.save_path, output_fname='test.txt', exp_name=config.experiment_name)

        model_test(net, config, test_data, task_func, logger)
        logger.dump_tabular()

    log_complete(config.save_path, start_time)