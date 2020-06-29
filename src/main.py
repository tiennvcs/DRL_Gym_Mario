"""Super Mario Bros for OpenAI Gym."""
import gym
from nes_py.wrappers import JoypadSpace
from actions import ACTION_SPACES
from play_dqn import play_dqn
from config import parameters
from utils import get_args

def main():
    """The main entry point for the command line interface."""
    # parse arguments from the command line (argparse validates arguments)
    args = get_args()
    # build the environment with the given ID
    env = gym.make(args.env)
    # wrap the environment with an action space if specified
    env = JoypadSpace(env, ACTION_SPACES[args.actionspace])
    # Play game/ Training agent
    play_dqn(env, parameters["parameters_" + str(args.p)])

if __name__ == '__main__':
    main()
