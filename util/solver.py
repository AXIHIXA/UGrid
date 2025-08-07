import typing

import numpy as np
# import pyamgcl
import scipy.sparse
import scipy.sparse.linalg


def assemble_poisson_problem(bc_value: np.ndarray,
                             bc_mask: np.ndarray,
                             f: typing.Optional[np.ndarray]) \
        -> typing.Tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """
    Assemble sparse matrix A for Possion problem A x = rhs.
    There will be an "testcase-concept" 1-width padding around the rectangular grid (trivial zero boundary condition),
    but this padding will not appear testcase A or rhs (thus does not affect the problem size).
    :param bc_mask:
    :param bc_value:
    :param f:
    :return: A, rhs
    """
    image_size: int = bc_mask.shape[0]
    matrix_size: int = image_size * image_size  # == 1/h^2

    row: typing.List[int] = []
    col: typing.List[int] = []
    data: typing.List[float] = []

    k: int = 0  # maintained inside the loop as k == i * image_size + j

    for i in range(image_size):
        for j in range(image_size):
            if bc_mask[i, j] == 1:
                # boundary pixels
                row.append(k)
                col.append(k)
                data.append(1)

            else:
                # five-point stencil for interior pixels
                # i, j - 1
                if j != 0:
                    row.append(k)
                    col.append(k - 1)
                    data.append(1)
                # i, j + 1
                if j != image_size - 1:
                    row.append(k)
                    col.append(k + 1)
                    data.append(1)
                # i - 1, j
                if i != 0:
                    row.append(k)
                    col.append(k - image_size)
                    data.append(1)
                # i + 1, j
                if i != image_size - 1:
                    row.append(k)
                    col.append(k + image_size)
                    data.append(1)
                # i, j
                row.append(k)
                col.append(k)
                data.append(-4)

            k += 1

    A = scipy.sparse.csr_matrix((data, (row, col)), shape=(matrix_size, matrix_size), dtype=np.float32)

    rhs: np.ndarray = bc_value.reshape(-1)

    if f is not None:
        rhs += f.reshape(-1)

    return A, rhs


# # noinspection DuplicatedCode, PyPep8Naming
def scipy_solve(bc_value: np.ndarray,
                bc_mask: np.ndarray,
                f: typing.Optional[np.ndarray]):
    A, rhs = assemble_poisson_problem(bc_value, bc_mask, f)
    X = scipy.sparse.linalg.spsolve(A, rhs)
    X.resize(bc_mask.shape)
    return X.astype(np.float32)


# # noinspection DuplicatedCode, PyPep8Naming
# def amgcl_solve(bc_value: np.ndarray,
#                 bc_mask: np.ndarray,
#                 f: typing.Optional[np.ndarray],
#                 rel_tol: float = 1e-4) \
#         -> np.ndarray:
#     """
#     Solve a discrete 2D Poisson problem on uniform square grid with pyamgcl https://github.com/ddemidov/amgcl.
#             (1 - bc_mask) A x = (1 - bc_mask) f
#                  bc_mask    x =      bc_mask  bc_value
#     For simplicity:
#                    f == (1 - bc_mask) * f
#             bc_value == (1 - bc_mask) * bc_value
#     There will be an "testcase-concept" 1-width padding around the rectangular grid (trivial zero boundary condition),
#     but this padding will not appear testcase A or rhs (thus does not affect the problem size).
#     :param bc_mask:
#     :param bc_value:
#     :param f:
#     :param rel_tol:
#     :return: solution
#     """
#     A, rhs = assemble_poisson_problem(bc_value, bc_mask, f)
#
#     P = pyamgcl.amg(A, prm={'relax.type': 'spai0'})
#     solver = pyamgcl.solver(P, prm={'type': 'bicgstab', 'tol': rel_tol, 'abstol': rel_tol * np.linalg.norm(rhs)})
#
#     x: np.ndarray = solver(rhs)
#     x.resize(bc_mask.shape)
#
#     return x
#
#
# # noinspection DuplicatedCode, PyPep8Naming
# def amgcl_solve_post_assembly(A: scipy.sparse.csr_matrix,
#                               rhs: np.ndarray,
#                               rel_tol: float = 1e-4) \
#         -> np.ndarray:
#     """
#     Solve a discrete 2D Poisson problem on uniform square grid with pyamgcl https://github.com/ddemidov/amgcl.
#             A x = rhs
#     :param A:
#     :param rhs:
#     :param rel_tol:
#     :return: solution (viewed as 1D vector)
#     """
#     P = pyamgcl.amg(A, prm={'relax.type': 'spai0'})
#     solver = pyamgcl.solver(P, prm={'type': 'bicgstab',
#                                     'tol': rel_tol,
#                                     'abstol': rel_tol * np.linalg.norm(rhs),
#                                     'maxiter': 100})
#
#     x: np.ndarray = solver(rhs)
#
#     return x
