import argparse

def get_args():
    """Parse command line arguments and return them."""
    parser = argparse.ArgumentParser(description="Deep Q Lerning in Mario Bros Game")
    # add the argument for the Super Mario Bros environment to run
    parser.add_argument('--env', '-e',
        type=str,
        default='SuperMarioBros-v0',
        help='The name of the environment to play'
    )

    # add parameter selections
    parser.add_argument('--parameter', '-p',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help='The parameter to train agent'
    )
    # add the argument for adjusting the action space
    parser.add_argument('--actionspace', '-a',
        type=str,
        default='simple',
        choices=['nes', 'right', 'simple', 'complex'],
        help='the action space wrapper to use'
    )
    # add the argument for the number of steps to take in random mode
    parser.add_argument('--steps', '-s',
        type=int,
        default=500,
        help='The number of random steps to take.',
    )
    # parse arguments and return them
    return parser.parse_args()
