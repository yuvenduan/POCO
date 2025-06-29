import os
import os.path as osp
import torch.nn as nn
import logging

from torch.nn.utils import clip_grad_norm_
from configs.configs import BaseConfig
from tasks import taskfunctions
from datetime import datetime

def grad_clipping(model, max_norm, printing=False):
     
    p_req_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]

    if printing:
        grad_before = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_before += param_norm.item() ** 2
        grad_before = grad_before ** (1. / 2)

    clip_grad_norm_(p_req_grad, max_norm)

    if printing:
        grad_after = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_after += param_norm.item() ** 2
        grad_after = grad_after ** (1. / 2)

        if grad_before > grad_after:
            print("clipped")
            print("before: ", grad_before)
            print("after: ", grad_after)

def log_complete(exp_path: str, start_time=None):
    """
    create a file to indicate the training is finished
    """
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    complete_time = datetime.now()
    with open(osp.join(exp_path, 'train_complete.txt'), 'w') as f:
        f.write(f'Training is complete at: {complete_time.strftime("%Y-%m-%d %H:%M:%S")}')
        if start_time is not None:
            f.write(f'\nTraining time: {str(complete_time - start_time)}')
    
    logging.info(f'Training is complete at: {complete_time.strftime("%Y-%m-%d %H:%M:%S")}')


def get_grad_norm(model: nn.Module):
    g = 0
    for param in model.parameters():
        g += param.grad.square().sum()
    return g

def task_init(config_: BaseConfig, input_size=None):
    """initialize tasks"""
    task_type = config_.task_type

    if task_type == 'neural_prediction':
        task_func_ = taskfunctions.NeuralPrediction(config_, input_size=input_size)
    else:
        raise NotImplementedError('task not implemented')

    return task_func_
