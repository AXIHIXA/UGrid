import datetime

from .base_arg import BaseArg


class TrainArg(BaseArg):
    def __init__(self) -> None:
        super().__init__()

        # model
        self.parser.add_argument('--structure',
                                 type=str,
                                 required=True,
                                 choices=['unet'],
                                 help='specify network structure')
        self.parser.add_argument('--downsampling_policy',
                                 type=str,
                                 required=True,
                                 choices=['lerp', 'conv'],
                                 help='downsampling policy')
        self.parser.add_argument('--upsampling_policy',
                                 type=str,
                                 required=True,
                                 choices=['lerp', 'conv'],
                                 help='upsampling policy')
        self.parser.add_argument('--num_iterations',
                                 type=int,
                                 required=True,
                                 help='number of iterations to apply testcase the solver')
        self.parser.add_argument('--relative_tolerance',
                                 type=float,
                                 required=True,
                                 help='max relative residual error allowed')
        self.parser.add_argument('--initialize_x0',
                                 type=str,
                                 required=True,
                                 choices=['random', 'zero', 'avg'],
                                 help='method to initialize the inner part of the initial guess x_0')

        self.parser.add_argument('--num_mg_layers',
                                 type=int,
                                 required=True,
                                 help='number of layers testcase the multigrid method')
        self.parser.add_argument('--num_mg_pre_smoothing',
                                 type=int,
                                 required=True,
                                 help='number of pre-smoothing iterations testcase multigrid')
        self.parser.add_argument('--num_mg_post_smoothing',
                                 type=int,
                                 required=True,
                                 help='number of post-smoothing iterations testcase multigrid')

        self.parser.add_argument('--activation',
                                 type=str,
                                 nargs='+',
                                 required=True,
                                 help=r'Activation applied after each iteration. '
                                      r'Choices are "none", "clamp low high", "leaky_relu negative_slope"')

        # initialization
        self.parser.add_argument('--initialize_trainable_parameters',
                                 type=str,
                                 required=True,
                                 choices=['default'],
                                 help='method to initialize the trainable parameters')

        # optimization
        self.parser.add_argument('--optimizer',
                                 type=str,
                                 required=True,
                                 choices=['adam', 'rmsprop', 'sgd'])

        self.parser.add_argument('--scheduler',
                                 type=str,
                                 nargs='+',
                                 required=True,
                                 help='Scheduler used for training. Choices are "step step_size gamma"')
        self.parser.add_argument('--initial_lr',
                                 type=float,
                                 required=True,
                                 help='initial learning rate')

        self.parser.add_argument('--lambda_1',
                                 type=float,
                                 required=True,
                                 help='lamdba_1 for loss function')
        self.parser.add_argument('--lambda_2',
                                 type=float,
                                 required=True,
                                 help='lamdba_2 for loss function')

        # epoch numbers
        self.parser.add_argument('--start_epoch',
                                 type=int,
                                 required=True,
                                 help='starting epoch number')
        self.parser.add_argument('--max_epoch',
                                 type=int,
                                 required=True,
                                 help='total number of epochs')
        self.parser.add_argument('--save_every',
                                 type=int,
                                 required=True,
                                 help='dump current model every x epochs')
        self.parser.add_argument('--evaluate_every',
                                 type=int,
                                 required=True,
                                 help='evaluate on validation set every x epochs')

        # dataset
        self.parser.add_argument('--dataset_root',
                                 type=str,
                                 required=True,
                                 help='root directory containing train and evaluate dataset')
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 required=True,
                                 help='number of threads used testcase data loader')
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 required=True)

        # checkpoint
        self.parser.add_argument('--checkpoint_root',
                                 type=str,
                                 required=True,
                                 help='the directory to save/load checkpoints')
        self.parser.add_argument('--load_experiment',
                                 type=str,
                                 default=None,
                                 help='name of the experiment to load')
        self.parser.add_argument('--load_epoch',
                                 type=int,
                                 default=None,
                                 help='epoch of experiment to load, -1 means the latest')

        # reproducibility
        self.parser.add_argument('--seed',
                                 type=int,
                                 default=None,
                                 help='manual seed PyTorch and NumPy with this seed')
        self.parser.add_argument('--deterministic',
                                 default=False,
                                 action='store_true',
                                 help='apply this option to enforce deterministic algorithms')
