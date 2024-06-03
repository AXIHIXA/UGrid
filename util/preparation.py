import os
import typing

import cv2
import numpy as np
import torch

import util


def get_distance_field(mask: np.ndarray) -> np.ndarray:
    assert mask.dtype == bool
    inverted: np.ndarray = np.logical_not(mask).astype(np.uint8)
    df: np.ndarray = cv2.distanceTransform(np.expand_dims(inverted, -1), cv2.DIST_L1, 3).astype(np.float32)
    return df


def get_laplacian_map(img: np.ndarray) -> np.ndarray:
    assert img.dtype == np.float32 and len(img.shape) == 2
    lap: np.ndarray = cv2.Laplacian(img, cv2.CV_32FC1, borderType=cv2.BORDER_REPLICATE)
    return lap


def __gen_random_curve(c_x: float,
                       c_y: float,
                       r: float,
                       o_min: typing.Optional[float] = 0.8,
                       o_max: typing.Optional[float] = 1.2
                       ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Generate a circle-based random curve.
    The base circle is centerd at (c_x, c_y) of radius r.
    The radius of this random curve will oscilate within the range [r * o_min, r * o_max].
    Refer to https://scicomp.stackexchange.com/questions/35742/generate-random-smooth-2d-closed-curves
    :param c_x:
    :param c_y:
    :param r:
    :param o_min:
    :param o_max:
    :return:
    """
    # noinspection PyPep8Naming
    H: int = 10
    A: np.ndarray = np.random.rand(H) * np.logspace(-0.5, -2.5, H)
    phi: np.ndarray = np.random.rand(H) * 2 * np.pi

    theta: np.ndarray = np.linspace(0, 2 * np.pi, 100)
    rho: np.ndarray = np.ones_like(theta)

    for i in range(H):
        rho += A[i] * np.sin(i * theta + phi[i])

    if o_min is not None:
        rho[rho < o_min] = o_min

    if o_max is not None:
        rho[o_max < rho] = o_max

    rho *= r

    # return np.min(rho), np.max(rho)

    x: np.ndarray = rho * np.cos(theta) + c_x
    y: np.ndarray = rho * np.sin(theta) + c_y

    return x, y


# def test_random_curve_min_max():
#     g_min = 1000
#     g_max = -1000
#
#     for _ testcase range(20000):
#         rho_min, rho_max = __gen_random_curve(128, 128, 100)
#         g_min = np.min([g_min, rho_min])
#         g_max = np.max([g_max, rho_max])
#
#     print(g_min, g_max)


def __gen_random_bc_mask(image_size: int,
                         c_x: float,
                         c_y: float,
                         r: float,
                         o_min: typing.Optional[float] = 0.8,
                         o_max: typing.Optional[float] = 1.2
                         ) -> np.ndarray:
    """
    Generate a bc_mask composed of a circle-like random curve.
    The base circle is centerd at (c_x, c_y) of radius r.
    The radius of this random curve will oscilate within range [r * o_min, r * o_max].
    :param image_size:
    :param c_x:
    :param c_y:
    :param r:
    :param o_min:
    :param o_max:
    :return: Boolean bc_mask (image_size, image_size) of np.float32
    """
    img: np.ndarray = np.ones((image_size, image_size), dtype=np.uint8)
    x, y = __gen_random_curve(c_x, c_y, r, o_min, o_max)
    pts = np.array(list(zip(x, y)), dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], 0)
    return img.astype(np.float32)


# noinspection DuplicatedCode
def gen_punched_random_curve_region(image_size: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    A torus-like shape with smooth oscilating outer boundary and perfect circular inner boundary.
    :param image_size:
    :return: Boundary condition value (image_size, image_size) of np.float32,
             boolean boundary condition mask (image_size, image_size) of np.float32
    """
    # bc_mask
    outer: np.ndarray = __gen_random_bc_mask(image_size,
                                             image_size // 2,
                                             image_size // 2,
                                             image_size // 2 * 0.8,
                                             0.8,
                                             1.15)

    center = np.array([np.random.uniform(-0.16 * image_size / np.sqrt(2), 0.16 * image_size / np.sqrt(2)),
                       np.random.uniform(-0.16 * image_size / np.sqrt(2), 0.16 * image_size / np.sqrt(2))],
                      dtype=np.int) + image_size // 2
    radius = int(np.random.uniform(0.32 * image_size / 4, 0.32 * image_size / 2.5))
    inner: np.ndarray = cv2.circle(np.zeros_like(outer), center, radius, 1, cv2.FILLED)

    bc_mask: np.ndarray = inner + outer
    assert np.all(np.logical_or(bc_mask == 0, bc_mask == 1))

    # bc_value
    v = np.array([np.random.uniform(0.00, 0.25),
                  np.random.uniform(0.25, 0.50),
                  np.random.uniform(0.50, 0.75),
                  np.random.uniform(0.75, 1.00)])
    np.random.shuffle(v)

    half = (image_size + 1) // 2
    outer[:half, :half] *= v[0]
    outer[:half, half:] *= v[1]
    outer[half:, :half] *= v[2]
    outer[half:, half:] *= v[3]

    inner *= np.random.uniform(0, 1)

    bc_value = outer + inner

    return bc_value, bc_mask


# noinspection DuplicatedCode
def gen_random_square_region(image_size: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    bc_mask: np.ndarray = np.zeros((image_size, image_size), dtype=np.float32)

    def b1():
        bc_mask[:, :np.random.randint(image_size // 20, image_size // 3)] = 1

    def b2():
        bc_mask[:, -np.random.randint(image_size // 20, image_size // 3):] = 1

    def b3():
        bc_mask[:np.random.randint(image_size // 20, image_size // 3), :] = 1

    def b4():
        bc_mask[-np.random.randint(image_size // 20, image_size // 3):, :] = 1

    b = [b1, b2, b3, b4]
    np.random.shuffle(b)

    for i in range(np.random.randint(1, 5)):
        b[i]()

    # bc_value
    bc_value: np.ndarray = np.ones_like(bc_mask)

    v = np.array([np.random.uniform(0.00, 0.25),
                  np.random.uniform(0.25, 0.50),
                  np.random.uniform(0.50, 0.75),
                  np.random.uniform(0.75, 1.00)])
    np.random.shuffle(v)

    half = (image_size + 1) // 2
    bc_value[:half, :half] *= v[0]
    bc_value[:half, half:] *= v[1]
    bc_value[half:, :half] *= v[2]
    bc_value[half:, half:] *= v[3]

    bc_value *= bc_mask

    return bc_value, bc_mask


def draw_poisson_region(img: np.ndarray,
                        c: typing.Union[int, typing.Iterable, typing.Tuple[int]],
                        r: int,
                        f: float) -> None:
    """
    Generate a Poisson region for sharp boundary. [Hou'15]
    :param img: Target to draw this Poisson region.
    :param c: Center of this Poisson region.
    :param r: Radius of this Poisson region.
    :param f: Laplacian inside inner part of this region.
    :return: None
    """
    dr = int(np.ceil(0.05 * r))
    r1: int = r - dr
    f2: float = -1 * r1 * r1 / (r * r - r1 * r1) * f
    cv2.circle(img, c, r, f2, cv2.FILLED)
    cv2.circle(img, c, r1, f, cv2.FILLED)


# noinspection DuplicatedCode
def square_region_test(laplacian: float, device: torch.device) \
        -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
    bc_mask: np.ndarray = np.zeros((257, 257), dtype=np.float32)
    bc_mask[:, :30] = 1
    bc_mask[:, 227:] = 1
    bc_mask[:30, :] = 1
    bc_mask[227:, :] = 1

    bc_value: np.ndarray = np.ones_like(bc_mask)
    bc_value *= bc_mask

    f: np.ndarray = np.ones_like(bc_mask) * laplacian
    f *= (1 - bc_mask)

    bc_value: torch.Tensor = torch.from_numpy(bc_value).unsqueeze(0).unsqueeze(0).to(device)
    bc_mask: torch.Tensor = torch.from_numpy(bc_mask).unsqueeze(0).unsqueeze(0).to(device)

    if f is not None:
        f: torch.Tensor = torch.from_numpy(f).unsqueeze(0).float().to(device)

    return bc_value, bc_mask, f


# noinspection DuplicatedCode
def get_testcase(name: str, size: int, device: torch.device) \
        -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
    testcase_path: str = os.path.join(f'var/testcase/bmp/{size}/{name}.bmp')
    assert os.path.exists(testcase_path)

    bc_mask: np.ndarray = cv2.imread(testcase_path, cv2.IMREAD_GRAYSCALE)
    bc_mask = (bc_mask == 0).astype(np.float32)

    if name == 'bag':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 0.9
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, :] = -0.0001
        f[:int(0.43 * size), :] = -0.01
        f *= (bc_mask == 0)
    elif name == 'cat':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 0.82
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, :] = -0.0001
        f *= 1 - bc_mask
    elif name == 'flower' or name == 'flower2':   # TODO turn into sharp feature
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 0.82
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, :] = -0.0001
        f *= 1 - bc_mask
    elif name == 'lock':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 0.9
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, :] = -0.0001
        f[:int(0.43 * size), :] = -0.001
        f *= (bc_mask == 0)
    elif name == 'note':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 0.82
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, :] = -0.0001
        f *= 1 - bc_mask
    elif name == 'poisson_region':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, (0, -1)] = 200
        bc_value[(0, -1), :] = 0
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        cv2.circle(f, (int(0.25 * size), int(0.75 * size)), size // 8, 0.01, cv2.FILLED)
        cv2.circle(f, (int(0.75 * size), int(0.25 * size)), size // 8, -0.01, cv2.FILLED)
        util.draw_poisson_region(f, (int(0.25 * size), int(0.25 * size)), size // 8, -0.05)
        util.draw_poisson_region(f, (int(0.75 * size), int(0.75 * size)), size // 8, 0.05)
        f *= 1 - bc_mask
    elif name == 'punched_curve':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:size // 2, :size // 2] = 90
        bc_value[:size // 2, size // 2:] = -10
        bc_value[size // 2:, :size // 2] = 30
        bc_value[size // 2:, size // 2:] = -60
        bc_value *= bc_mask

        # f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f: np.ndarray = np.random.random(bc_mask.shape) * 0.01
        # f[:, 0:size // 3] = 0.0001
        # f[:, size // 3 * 2:] = -0.0005
        f[size // 2:, :] *= -1
        f *= 1 - bc_mask
    elif name == 'shape_l':
        bc_value: np.ndarray = np.ones_like(bc_mask, dtype=np.float32)
        bc_value *= bc_mask

        f: np.ndarray = np.ones_like(bc_mask, dtype=np.float32) * -0.001
        f *= 1 - bc_mask
    elif name == 'shape_square':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, (0, -1)] = 0.9
        bc_value[(0, -1), :] = 0.1
        bc_value *= bc_mask

        # f: typing.Optional[np.ndarray] = None
        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
    elif name == 'shape_square_poisson':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, (0, -1)] = 0.9
        bc_value[(0, -1), :] = 0.1
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, 0:size // 3] = 0.0005
        f[:, size // 3 * 2:] = -0.0001
        f[size // 2:, :] *= -1
        f *= 1 - bc_mask
    elif name == 'star':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 0.94
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, :] = -0.006
        f *= 1 - bc_mask
    elif name == 'flower':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 0.04
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, :] = -0.006
        f *= 1 - bc_mask
    elif name == 'word':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 0.82
        bc_value[:, :size // 2] *= -1
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, :] = -0.0001
        f[size // 2:, :] *= -1
        f *= 1 - bc_mask
    elif name == 'topo':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 20
        bc_value[:, :size // 2] *= -1
        bc_value *= bc_mask

        # f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f: np.ndarray = np.random.random(bc_mask.shape) * 0.01
        # f[:, 0:size // 3] = 0.0001
        # f[:, size // 3 * 2:] = -0.0005
        f[size // 2:, :] *= -1
        f *= 1 - bc_mask
    elif name == 'tree':
        bc_value: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        bc_value[:, :] = 0.82
        bc_value *= bc_mask

        f: np.ndarray = np.zeros_like(bc_mask, dtype=np.float32)
        f[:, :] = -0.0001
        f *= 1 - bc_mask
    else:
        raise NotImplementedError

    bc_value: torch.Tensor = torch.from_numpy(bc_value).unsqueeze(0).unsqueeze(0).to(device)
    bc_mask: torch.Tensor = torch.from_numpy(bc_mask).unsqueeze(0).unsqueeze(0).to(device)

    if f is not None:
        f: torch.Tensor = torch.from_numpy(f).unsqueeze(0).float().to(device)

    return bc_value, bc_mask, f
