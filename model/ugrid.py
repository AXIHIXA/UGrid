import time
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F

import util


class UGrid(torch.nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_pre_smoothing: int,
                 num_post_smoothing: int,
                 downsampling_policy: str,
                 upsampling_policy,
                 activation: str,
                 initialize_trainable_parameters: str):
        super().__init__()

        self.num_layers: int = num_layers
        self.num_pre_smoothing: int = num_pre_smoothing
        self.num_post_smoothing: int = num_post_smoothing

        self.downsampling_policy: str = downsampling_policy
        self.upsampling_policy: str = upsampling_policy
        self.activation: str = activation
        self.initialize_trainable_parameters: str = initialize_trainable_parameters

        # Multigrid layer (UNet skip-connection blocks)
        self.mg = UNetSkipConnectionBlock(self.num_pre_smoothing,
                                          self.num_post_smoothing,
                                          self.downsampling_policy,
                                          self.upsampling_policy,
                                          self.initialize_trainable_parameters,
                                          None)

        for _ in range(self.num_layers - 1):
            self.mg = UNetSkipConnectionBlock(self.num_pre_smoothing,
                                              self.num_post_smoothing,
                                              self.downsampling_policy,
                                              self.upsampling_policy,
                                              self.initialize_trainable_parameters,
                                              self.mg)

        # Activation layer
        if 'none' in self.activation:
            self.activate = None
        elif 'clamp' in self.activation:
            _, low, high = self.activation
            self.activate = lambda x: x.clamp(float(low), float(high))
        elif 'leaky_relu' in self.activaion:
            _, negative_slope = self.activation
            self.activate = lambda x: F.leaky_relu(x, float(negative_slope))
        else:
            raise NotImplementedError

    # # TODO: Comment in benchmark code!
    # # FOR DUMPING INTERMEDIATE RESULTS, DO NOT USE FOR BENCHMARKS!
    # def forward(self,
    #             x: torch.Tensor,
    #             bc_value: torch.Tensor,
    #             bc_mask: torch.Tensor,
    #             f: typing.Optional[torch.Tensor]
    #             ) -> torch.Tensor:
    #     """
    #     Masked discrete Poisson equation (with arbitray Dirchilet boundary condition):
    #             (I - bc_mask) A x = (I - bc_mask) f
    #                  bc_mask    x =        bc_value
    #     Note for simplicity,
    #         we preprocess bc_value s.t. bc_value == bc_mask bc_value,
    #         and we have the exterior band of x be zero (trivial boundary condition).
    #
    #     Masked Jacobi update: Step matrix P = -4I
    #             x' = (I - bc_mask) ( (I - P^-1 A) x                        + P^-1 f ) + bc_value
    #                = (I - bc_mask) ( F.conv2d(x, jacobi_kernel, padding=1) - 0.25 f ) + bc_value
    #                = util.jacobi_step(x)
    #     """
    #     y = x
    #
    #     timestamp: str = '{}'.format(time.perf_counter_ns())
    #
    #     dic: typing.Dict[str, np.ndarray] = {}
    #     dic[f'1-x-before-presmooth'] = y.detach().squeeze().cpu().numpy()
    #
    #     for _ in range(self.num_pre_smoothing):
    #         y = util.jacobi_step(y, bc_value, bc_mask, f)
    #
    #     dic[f'2-x-after-presmooth'] = y.detach().squeeze().cpu().numpy()
    #
    #     r = util.absolute_residue(y, bc_mask, f, reduction='none').view_as(y)
    #
    #     dic[f'3-residue'] = r.detach().squeeze().cpu().numpy()
    #
    #     # fig = plt.figure(figsize=(5, 5), dpi=100)
    #     # plt.axis('off')
    #     # plt.imshow(r.detach().squeeze().cpu().numpy(), vmin=-0.1, vmax=0.01)
    #     # plt.savefig(f'var/out/{timestamp}_residue.png', bbox_inches='tight')
    #     # plt.close(fig)
    #
    #     r = self.mg(r, 1 - bc_mask)  # interior mask here
    #
    #     dic[f'4-delta'] = r.detach().squeeze().cpu().numpy()
    #
    #     y = y + r
    #
    #     dic[f'5-x+delta'] = y.detach().squeeze().cpu().numpy()
    #
    #     for _ in range(self.num_post_smoothing):
    #         y = util.jacobi_step(y, bc_value, bc_mask, f)
    #
    #     dic[f'6-x-after-postsmooth'] = y.detach().squeeze().cpu().numpy()
    #
    #     for name, img in dic.items():
    #         np.save(f'var/cr/{timestamp}_{name}.npy', img)
    #         util.plt_dump({f'{timestamp}_{name}': img}, dump_dir=f'var/cr/')
    #
    #     if self.activate is not None:
    #         y = self.activate(y)
    #
    #     return y

    # ORIGINAL VERSION FOR EFFICIENCY BENCHMARKING.
    def forward(self,
                x: torch.Tensor,
                bc_value: torch.Tensor,
                bc_mask: torch.Tensor,
                f: typing.Optional[torch.Tensor]
                ) -> torch.Tensor:
        """
        Masked discrete Poisson equation (with arbitray Dirchilet boundary condition):
                (I - bc_mask) A x = (I - bc_mask) f
                     bc_mask    x =        bc_value
        Note for simplicity,
            we preprocess bc_value s.t. bc_value == bc_mask bc_value,
            and we have the exterior band of x be zero (trivial boundary condition).

        Masked Jacobi update: Step matrix P = -4I
                x' = (I - bc_mask) ( (I - P^-1 A) x                        + P^-1 f ) + bc_value
                   = (I - bc_mask) ( F.conv2d(x, jacobi_kernel, padding=1) - 0.25 f ) + bc_value
                   = util.jacobi_step(x)
        """
        y = x

        timestamp: str = '{}'.format(time.perf_counter_ns())
        #
        # dic: typing.Dict[str, np.ndarray] = {}
        #
        # dic[f'1-x-before-presmooth'] = y.detach().squeeze().cpu().numpy()

        for _ in range(self.num_pre_smoothing):
            y = util.jacobi_step(y, bc_value, bc_mask, f)

        # dic[f'2-x-after-presmooth'] = y.detach().squeeze().cpu().numpy()

        r = util.absolute_residue(y, bc_mask, f, reduction='none').view_as(y)

        # residue_np: np.ndarray = r.detach().squeeze().cpu().numpy()
        # dic[f'3-residue'] = r.detach().squeeze().cpu().numpy()

        # fig = plt.figure(figsize=(5, 5), dpi=100)
        # plt.axis('off')
        # plt.imshow(r.detach().squeeze().cpu().numpy(), vmin=-0.1, vmax=0.01)
        # plt.savefig(f'var/out/{timestamp}_residue.png', bbox_inches='tight')
        # plt.close(fig)

        r = self.mg(r, 1 - bc_mask)  # interior mask here

        # dic[f'4-delta'] = r.detach().squeeze().cpu().numpy()

        y = y + r

        # dic[f'5-x+delta'] = y.detach().squeeze().cpu().numpy()

        for _ in range(self.num_post_smoothing):
            y = util.jacobi_step(y, bc_value, bc_mask, f)

        # dic[f'6-x-after-postsmooth'] = y.detach().squeeze().cpu().numpy()

        # util.plt_dump(dic, dump_dir=f'var/out/{timestamp}', colorbar=False)
        # util.plt_subplot(dic, suptitle=f'{timestamp}', )

        if self.activate is not None:
            y = self.activate(y)

        return y


class UNetSkipConnectionBlock(torch.nn.Module):
    """
    Defines the UNet submodule with skip connection as follows:
        x ------------------ identity ----------------- y
        |-- downsampling --- submodule --- upsampling --|
    Implements the recursive multigrid hierarchy.
    """
    def __init__(self,
                 num_pre_smoothing: int,
                 num_post_smoothing: int,
                 downsampling_policy: str,
                 upsampling_policy: str,
                 initialize_trainable_parameters: str,
                 submodule: typing.Optional[torch.nn.Module]):
        super().__init__()

        self.num_pre_smoothing: int = num_pre_smoothing
        self.num_post_smoothing: int = num_post_smoothing
        self.downsampling_policy: str = downsampling_policy
        self.upsampling_policy: str = upsampling_policy
        self.initialize_trainable_parameters: str = initialize_trainable_parameters

        self.submodule: typing.Optional[torch.nn.Module] = submodule

        # Convolution (linear) layers
        self.pre_smoothers = torch.nn.ModuleList(
                [torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False) for _ in range(self.num_pre_smoothing)])

        if self.downsampling_policy == 'lerp':
            self.downsampler = util.downsample2x
        elif self.downsampling_policy == 'conv':
            self.downsampler = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1, bias=False)
        else:
            raise NotImplementedError

        if self.upsampling_policy == 'lerp':
            self.upsampler = util.upsample2x
        elif self.upsampling_policy == 'conv':
            self.upsampler = torch.nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, bias=False)
        else:
            raise NotImplementedError

        self.post_smoothers = torch.nn.ModuleList(
                [torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False) for _ in range(self.num_post_smoothing)])

        # Initialization of trainable parameters
        if self.initialize_trainable_parameters == 'default':
            # PyTorch's default initialization.
            # For ConvNd, is kaiming initialization with a = sqrt(5).
            self.initialize: typing.Optional[typing.Callable[[torch.Tensor], None]] = None
        else:
            raise NotImplementedError

        if self.initialize is not None:
            for module in self.pre_smoothers:
                self.initialize(module.weight)

            if self.downsampling_policy == 'conv':
                self.initialize(self.downsampler.weight)

            if self.upsampling_policy == 'conv':
                self.initialize(self.upsampler.weight)

            for module in self.post_smoothers:
                self.initialize(module.weight)

    def forward(self, x: torch.Tensor, interior_mask: torch.Tensor) -> torch.Tensor:
        for pre_smoother in self.pre_smoothers:
            x = pre_smoother(x)
            x = x * interior_mask

        # Skip-connection
        if self.submodule is not None:
            x0 = x
            interior_mask_2h = self.downsampler(interior_mask)
            x = self.downsampler(x)
            x = self.submodule(x, interior_mask_2h)
            x = self.upsampler(x)
            x = x * interior_mask
            x = x + x0

        for post_smoother in self.post_smoothers:
            x = post_smoother(x)
            x = x * interior_mask

        return x
