"""Super Mario Bros for OpenAI Gym."""
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from actions import ACTION_SPACES
from play_dqn import play_dqn
from config import parameters
from utils import get_args, print_info_hyperparameters

def main():
    """The main entry point for the command line interface."""
    # parse arguments from the command line (argparse validates arguments)
    args = get_args()
    # build the environment with the given ID
    env = gym_super_mario_bros.make(args.env)
    # wrap the environment with an action space if specified
    env = JoypadSpace(env, ACTION_SPACES[args.actionspace])
    # Get the paramters and confirm
    parameters_x = parameters["parameters_" + str(args.parameter)]
    print_info_hyperparameters(parameters_x)

    ans = "Tien Dep Trai"

    # Commend these lines when training on server
    while not (ans.upper() == 'Y' or ans.upper() == 'YES' or ans.upper() == 'NO' or ans.upper() == 'N'):
        ans = input("\n ------------>  Do you want to continue with the above hyper-parameters ? (Y/N)")

    if ans.upper() == 'N' or ans.upper() == 'NO':
        print("Thank you and try again!")
        exit(0)

    # Training/play the agent with the hyper-parameters
    play_dqn(env, parameters_x)

if __name__ == '__main__':
    main()
