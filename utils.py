import argparse
import cv2
from config import STANDARD_SIZE

def get_args():
    """Parse command line arguments and return them."""
    parser = argparse.ArgumentParser(description="Deep Q Lerning in Mario Bros Game")
    # add the argument for the Super Mario Bros environment to run
    parser.add_argument('--env', '-e',
        type=str,
        default='SuperMarioBros-v0',
        help='The name of the environment to play'
    )

    parser.add_argument('--mode', '-m',
        type=str,
        default='train',
        choices=['train', 'play'],
        help='Give agent play game or Train agent'
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

    # add the argument for adjusting the action space
    parser.add_argument('--render', '-r',
        type=int,
        default=1,
        choices=[0, 1],
        help='Render the current state or not (0: no, 1: yes)'
    )
    # parse arguments and return them
    return parser.parse_args()

def print_info_hyperparameters(parameters_x):
    print("\n")
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

def render_frame(frame, ratio):
    width = frame.shape[1] * ratio
    height = frame.shape[0] * ratio
    dim = (width, height)
    resized_frame = cv2.cvtColor(cv2.resize(frame, dim, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    cv2.imshow('Mario game', resized_frame)