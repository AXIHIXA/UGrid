import argparse


class BaseArg:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self) -> argparse.Namespace:
        opt: argparse.Namespace = self.parser.parse_args()
        return opt
