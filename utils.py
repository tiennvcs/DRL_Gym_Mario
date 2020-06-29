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
    # parse arguments and return them
    return parser.parse_args()

def print_info_hyperparameters(parameters_x):
    print("The hyper-parameters information".center(100))
    print("-{1}-".format("|", "-"*98))
    print("| {0:40} | {1:54}|".format("Number of action", parameters_x['NUM_ACTIONS']))
    print("{0}{1}{0}".format("|", "-"*98))
    print("| {0:40} | {1:54}|".format("The gamma value", parameters_x['GAMMA']))
    print("{0}{1}{0}".format("|", "-"*98))
    print("| {0:40} | {1:54}|".format("The OBSERVE times", parameters_x['OBSERVE']))
    print("{0}{1}{0}".format("|", "-"*98))
    print("| {0:40} | {1:54}|".format("The EXPLORE times", parameters_x['EXPLORE']))
    print("{0}{1}{0}".format("|", "-"*98))
    print("| {0:40} | {1:54}|".format("The INITIAL_EPSILON value", parameters_x['INITIAL_EPSILON']))
    print("{0}{1}{0}".format("|", "-"*98))
    print("| {0:40} | {1:54}|".format("The FINAL_EPSILON value", parameters_x['FINAL_EPSILON']))
    print("{0}{1}{0}".format("|", "-"*98))
    print("| {0:40} | {1:54}|".format("The REPLAY MEMORY numbers", parameters_x['REPLAY_MEMORY']))
    print("{0}{1}{0}".format("|", "-"*98))
    print("| {0:40} | {1:54}|".format("The BATCH numbers", parameters_x['BATCH']))
    print("{0}{1}{0}".format("|", "-"*98))
    print("| {0:40} | {1:54}|".format("The FRAME_PER_ACTION numbers", parameters_x['FRAME_PER_ACTION']))
    print("-{1}-".format("|", "-"*98))
