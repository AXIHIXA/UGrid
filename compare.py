import shutil
import typing

import cv2
import numpy as np
import torch

import comparasion.cpmg as cpmg
import util


# noinspection DuplicatedCode
def amgcl_solve_on_single_data(testcase, size):
    b, m, f = util.get_testcase(testcase, size, util.get_device())
    x = util.initial_guess(b, m, 'random')
    b_np = b.cpu().squeeze().numpy()
    m_np = m.cpu().squeeze().numpy()
    f_np = f.cpu().squeeze().numpy()
    x_np = x.cpu().squeeze().numpy()

    assembly_time_elapsed: float = 0.0
    solving_time_elapsed: float = 0.0
    total_time_elapsed: float = 0.0
    total_error: float = 0.0
    duplicate: int = 10

    for _ in range(duplicate):
        iters, error, assembly_time, solving_time, total_time, y = cpmg.amgcl_solve(m_np, f_np, b_np, x_np, 1e-4)
        assembly_time_elapsed += assembly_time
        solving_time_elapsed += solving_time
        total_time_elapsed += total_time
        total_error += error

    log_str = (f'AMGCL CUDA {testcase}_{size} '
               f'assembly {assembly_time_elapsed / duplicate} '
               f'solving {solving_time_elapsed / duplicate} '
               f'total {total_time_elapsed / duplicate} '
               f'ms, rel res {total_error / duplicate}')
    print(log_str)
    # util.plt_subplot(
    #         dic={'m': m_np, 'f': f_np, 'b': b_np, 'y': y.reshape((size, size))},
    #         suptitle=log_str,
    #         show=False,
    #         dump=f'var/cmp_viz/{log_str}.png'
    # )


# noinspection DuplicatedCode
def amgx_solve_on_single_data(testcase, size):
    b, m, f = util.get_testcase(testcase, size, util.get_device())
    x0 = util.initial_guess(b, m, 'random')

    b_np: np.ndarray = b.detach().cpu().squeeze().numpy()
    m_np: np.ndarray = m.detach().cpu().squeeze().numpy()
    f_np: np.ndarray = f.detach().cpu().squeeze().numpy()
    x0_np: np.ndarray = x0.detach().cpu().squeeze().numpy()

    duplicate: int = 20
    assembly_time_elapsed: float = 0.0
    solving_time_elapsed: float = 0.0
    total_time_elapsed: float = 0.0
    total_error: float = 0.0

    for _ in range(duplicate):
        iters, assembly_time, solving_time, total_time, y = cpmg.amgx_solve(m_np, f_np, b_np, x0_np)
        assembly_time_elapsed += assembly_time
        solving_time_elapsed += solving_time
        total_time_elapsed += total_time
        total_error += util.relative_residue(torch.from_numpy(y.reshape((1, 1, size, size))).cuda(), b, m, f)[1].item()

    log_str = (f'AMGX {testcase}_{size} '
               f'assembly {assembly_time_elapsed / duplicate} '
               f'solving {solving_time_elapsed / duplicate} '
               f'total {total_time_elapsed / duplicate} '
               f'ms, rel res {total_error / duplicate}')
    print(log_str)
    # util.plt_subplot(
    #         dic={'m': m_np, 'f': f_np, 'b': b_np, 'y': y.reshape((size, size))},
    #         suptitle=log_str,
    #         show=False,
    #         dump=f'var/cmp_viz/{log_str}.png'
    # )


def main() -> None:
    # Same seed as the UGrid model for fair comparasion
    seed = 9590589012167207234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    testcase_lst: typing.List[str] = ['bag', 'cat', 'lock', 'note', 'poisson_region', 'punched_curve',
                                      'shape_l', 'shape_square', 'shape_square_poisson', 'star']

    # Test AMGCL CUDA
    for size in [1025, 257]:
        for testcase in testcase_lst:
            # print(f'AMGCL CUDA {testcase} {size}')
            # open(f'var/conv/AMGCL/{testcase}_{size}.txt', 'w').close()
            amgcl_solve_on_single_data(testcase, size)

    # Test NVIDIA AMGX
    cpmg.amgx_initialize()
    for size in [1025, 257]:
        for testcase in testcase_lst:
            amgx_solve_on_single_data(testcase, size)
    cpmg.amgx_finalize()


if __name__ == '__main__':
    main()














