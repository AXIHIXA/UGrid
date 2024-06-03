import argparse
import datetime
import os

import numpy as np
from loguru import logger
import torch
import torch.backends.cudnn

from arg import TrainArg
import model
import util


# noinspection DuplicatedCode
def main() -> None:
    # args
    opt: argparse.Namespace = TrainArg().parse()

    # checkpoint
    experienment_name: str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    experienment_checkpoint_path: str = os.path.join(opt.checkpoint_root, experienment_name)
    os.makedirs(experienment_checkpoint_path, exist_ok=True)
    np.save(os.path.join(experienment_checkpoint_path, 'opt_old.npy'), opt)

    # logger
    logger.add(os.path.join(experienment_checkpoint_path, 'train.log'))
    logger.info('======================== Args ========================')
    for k, v in vars(opt).items():
        logger.info(f'{k}\t\t{v}')
    logger.info('======================================================\n')

    # backend
    device: torch.device = util.get_device()
    logger.info(f'[Train] Using device {device}')

    if opt.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'[Train] Enforce deterministic algorithms, cudnn benchmark disabled')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(f'[Train] Do not enforce deterministic algorithms, cudnn benchmark enabled')

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        logger.info(f'[Train] Manual seed PyTorch with seed {opt.seed}\n')
    else:
        seed: int = torch.seed()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f'[Train] Using random seed {seed} for PyTorch\n')

    # torch.autograd.set_detect_anomaly(True)  # for debugging only, do NOT use testcase training!

    # training
    solver = model.Solver(opt.structure, opt.downsampling_policy, opt.upsampling_policy, device,
                          opt.num_iterations, opt.relative_tolerance, opt.initialize_x0,
                          opt.num_mg_layers, opt.num_mg_pre_smoothing, opt.num_mg_post_smoothing,
                          opt.activation, opt.initialize_trainable_parameters)
    trainer = model.Trainer(experienment_name, experienment_checkpoint_path, device,
                            solver, logger,
                            opt.optimizer, opt.scheduler, opt.initial_lr, opt.lambda_1, opt.lambda_2,
                            opt.start_epoch, opt.max_epoch, opt.save_every, opt.evaluate_every,
                            opt.dataset_root, opt.num_workers, opt.batch_size)
    trainer.train()


if __name__ == '__main__':
    main()
