from .base_arg import BaseArg


class TestArg(BaseArg):
    def __init__(self) -> None:
        super().__init__()

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
                                 help='name of the experiment to load')
        self.parser.add_argument('--load_epoch',
                                 type=int,
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