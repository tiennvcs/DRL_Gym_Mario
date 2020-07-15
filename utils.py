import argparse

def get_args():
    """Parse command line arguments and return them."""
    parser = argparse.ArgumentParser(description="Deep  double Q Lerning in Mario Bros Game")
    # add the argument for the Super Mario Bros environment to run
    parser.add_argument('--env', '-e',
        type=str,
        default='SuperMarioBros-v0',
        help='The name of the environment to play'
    )
    parser.add_argument('--n_replay', '-n',
        type=int,
        default=10,
        help='The numbers replay models '
    )
    

    
    # parse arguments and return them
    return parser.parse_args()

