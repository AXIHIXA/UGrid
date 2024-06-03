import argparse
import shutil
import time
import typing
import os

from loguru import logger
import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data

from arg import TestArg
from data import SynDat
from model import Solver
import util


# noinspection DuplicatedCode
def test_on_dataset(model, test_loader, device) -> None:
    with torch.no_grad():
        test_loss_dict: typing.Dict[str, typing.List[torch.Tensor]] = {}

        for batch in test_loader:
            x: typing.Optional[torch.Tensor] = None
            bc_value: torch.Tensor = batch['bc_value'].to(device)
            bc_mask: torch.Tensor = batch['bc_mask'].to(device)
            f: typing.Optional[torch.Tensor] = None

            tup: typing.Tuple[torch.Tensor, int] = model(x, bc_value, bc_mask, f)
            y, iterations_used = tup

            absolute_loss, relative_loss = util.relative_residue(y, bc_value, bc_mask, f)
            absolute_loss = absolute_loss.mean()
            relative_loss = relative_loss.mean()

            iterations_used = torch.tensor([iterations_used], dtype=torch.float32).to(device)

            if 'absolute_loss' in test_loss_dict:
                test_loss_dict['absolute_loss'].append(absolute_loss)
            else:
                test_loss_dict['absolute_loss']: typing.List[torch.Tensor] = [absolute_loss]

            if 'relative_loss' in test_loss_dict:
                test_loss_dict['relative_loss'].append(relative_loss)
            else:
                test_loss_dict['relative_loss']: typing.List[torch.Tensor] = [relative_loss]

            if 'iterations_used' in test_loss_dict:
                test_loss_dict['iterations_used'].append(iterations_used)
            else:
                test_loss_dict['iterations_used']: typing.List[torch.Tensor] = [iterations_used]

        for k, v in test_loss_dict.items():
            logger.info('[Test] {} = {}'.format(k, torch.mean(torch.tensor(v))))


# noinspection DuplicatedCode
def test_on_single_data(testcase: str,
                        size: int,
                        model: Solver,
                        device: torch.device,
                        benchmark_iteration: typing.Optional[int] = None) \
        -> None:
    with torch.no_grad():
        test_loss_dict: typing.Dict[str, typing.List[torch.Tensor]] = {}

        image_size: int = size
        bc_value, bc_mask, f = util.get_testcase(testcase, image_size, device)

        time_lst: typing.List[float] = []

        if benchmark_iteration is None:
            benchmark_iteration = 1

        for _ in range(benchmark_iteration):
            start_time: float = time.perf_counter_ns()
            tup: typing.Tuple[torch.Tensor, int] = model(None, bc_value, bc_mask, f)
            y, iterations_used = tup
            time_lst.append(time.perf_counter_ns() - start_time)

        # logger.info(f'[Test] Testcase {testcase} of size {size} min = {torch.min(y)} max = {torch.max(y)}')
        # np.save(f'var/testcase/npy/{testcase}_{size}.npy', np.array(time_lst))

        tup: typing.Tuple[torch.Tensor, torch.Tensor] = util.relative_residue(y, bc_value, bc_mask, f)
        abs_residual_norm, rel_residual_norm = tup
        abs_residual_norm: torch.Tensor = abs_residual_norm.mean()
        rel_residual_norm: torch.Tensor = rel_residual_norm.mean()

        iterations_used = torch.tensor([iterations_used], dtype=torch.float32).to(device)

        if 'abs_residual_norm' in test_loss_dict:
            test_loss_dict['abs_residual_norm'].append(abs_residual_norm)
        else:
            test_loss_dict['abs_residual_norm']: typing.List[torch.Tensor] = [abs_residual_norm]

        if 'rel_residual_norm' in test_loss_dict:
            test_loss_dict['rel_residual_norm'].append(rel_residual_norm)
        else:
            test_loss_dict['rel_residual_norm']: typing.List[torch.Tensor] = [rel_residual_norm]

        if 'iterations_used' in test_loss_dict:
            test_loss_dict['iterations_used'].append(iterations_used)
        else:
            test_loss_dict['iterations_used']: typing.List[torch.Tensor] = [iterations_used]

        # for k, v in test_loss_dict.items():
        #     logger.info('[Test] {} = {}'.format(k, torch.mean(torch.tensor(v))))

        bc_value_np: np.ndarray = bc_value.cpu().squeeze().numpy()
        bc_mask_np: np.ndarray = bc_mask.cpu().squeeze().numpy()
        f_np: typing.Optional[np.ndarray] = f.cpu().squeeze().numpy() if f is not None else None
        y_np: np.ndarray = y.cpu().squeeze().numpy()

        log_str = (f'UGrid {testcase}_{size} {np.mean(time_lst) / 1e6} ms, ' +
                   'rel res {}'.format(torch.mean(torch.tensor(test_loss_dict['rel_residual_norm'])).item()))
        logger.info(log_str)
        # util.plt_subplot(
        #         dic={'m': bc_mask_np, 'f': f_np, 'b': bc_value_np, 'y': y_np},
        #         suptitle=log_str,
        #         show=False,
        #         dump=f'var/cmp_viz/{log_str}.png'
        # )


# noinspection DuplicatedCode
def main() -> None:
    # argument parameters
    arg_opt: argparse.Namespace = TestArg().parse()

    # training parameters
    experiment_checkpoint_path: str = os.path.join(arg_opt.checkpoint_root, arg_opt.load_experiment)
    exp_opt_np: np.ndarray = np.load(os.path.join(experiment_checkpoint_path, 'opt.npy'), allow_pickle=True)

    # merged argument namespace
    opt: argparse.Namespace = util.merge_namespace(exp_opt_np.item(), arg_opt)

    logger.info('======================== Args ========================')
    for k, v in vars(opt).items():
        logger.info(f'{k}\t\t{v}')
    logger.info('======================================================\n')

    # backend
    device: torch.device = util.get_device()
    logger.info(f'[Test] Using device {device}')

    if opt.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'[Test] Enforce deterministic algorithms, cudnn benchmark disabled')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(f'[Test] Do not enforce deterministic algorithms, cudnn benchmark enabled')

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        logger.info(f'[Test] Manual seed PyTorch with seed {opt.seed}\n')
    else:
        seed: int = torch.seed()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f'[Test] Using random seed {seed} for PyTorch\n')

    # model
    opt.num_iterations = 64
    model = Solver(opt.structure, opt.downsampling_policy, opt.upsampling_policy, device,
                   opt.num_iterations, 1e-4, opt.initialize_x0,
                   opt.num_mg_layers, opt.num_mg_pre_smoothing, opt.num_mg_post_smoothing,
                   opt.activation, 'default')
    loaded_checkpoint = model.load(experiment_checkpoint_path, opt.load_epoch)
    model.eval()
    logger.info(f'[Test] Checkpoint loaded from {loaded_checkpoint}\n')

    # # test on dataset
    # # ##
    # test_dataset_path: str = os.path.join(opt.dataset_root, 'test')
    # test_dataset = SynDat(os.path.join(opt.dataset_root, 'test'))
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=opt.batch_size,
    #                                           num_workers=opt.num_workers,
    #                                           pin_memory=True)
    # logger.info(f'[Test] {len(test_dataset)} testing data loaded from {test_dataset_path}')
    # test_on_dataset(model, test_loader, device)

    testcase_lst: typing.List[str] = ['bag', 'cat', 'lock', 'note', 'poisson_region', 'punched_curve',
                                      'shape_l', 'shape_square', 'shape_square_poisson', 'star', 'bag']

    # Test UGrid
    # TODO: benchmark_iteration should be 100 for benchmarking.
    for size in [1025, 257]:
        for testcase in testcase_lst:
            # os.mkdir('var/conv/UGrid/tmp/')
            test_on_single_data(testcase, size, model, device, benchmark_iteration=10)
            # shutil.move('var/conv/UGrid/tmp/', f'var/conv/UGrid/{testcase}_{size}/')


if __name__ == '__main__':
    main()








