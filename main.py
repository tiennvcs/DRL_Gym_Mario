import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import tensorflow as tf

from actions import ACTION_SPACES
from agent import play_dqn, train_dqn
from config import parameters
from network_architecture import createNetwork
from utils import get_args, print_info_hyperparameters


def main():
    """The main entry point for the command line interface."""
    # parse arguments from the command line (argparse validates arguments)
    args = get_args()
    # build the environment with the given ID
    env = gym_super_mario_bros.make(args.env)
    # wrap the environment with an action space if specified
    env = JoypadSpace(env, ACTION_SPACES[args.actionspace])
    parameters_x = parameters["parameters_" + str(args.parameter)]

    # Training/play the agent with the hyper-parameters
    #################################################################################
    sess = tf.InteractiveSession()                                                  #
    # Initialize a neural network                                                   #
    s, readout, h_fc1 = createNetwork(parameters_x)                                 #                                                                      #
    # define the cost function                                                      #
    a = tf.placeholder("float", [None, len(ACTION_SPACES[args.actionspace])])       #
    y = tf.placeholder("float", [None])                                             #
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)    #
    cost = tf.reduce_mean(tf.square(y - readout_action))                            #
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)                        #
    #################################################################################

    saver = tf.train.Saver()                                                                   
    sess.run(tf.initialize_all_variables())                                                    

    if args.mode == 'train':
        # Get the paramters and confirm
        
        print_info_hyperparameters(parameters_x)

        ans = "Bao Tien Tri"

        # Comment these lines when training on server
        while not (ans.upper() == 'Y' or ans.upper() == 'YES' or ans.upper() == 'NO' or ans.upper() == 'N'):
            ans = input("\n ------------>  Do you want to continue with the above hyper-parameters ? (Y/N)")

        if ans.upper() == 'N' or ans.upper() == 'NO':
            print("Thank you and try again!")
            exit(0)
        train_dqn(env, parameters_x, readout=readout, s=s, saver=saver, sess=sess, train_step=train_step, h_fc1=h_fc1, render=args.render)

    elif args.mode == 'play':
        checkpoint = tf.train.get_checkpoint_state("saved_networks")                           
        if checkpoint and checkpoint.model_checkpoint_path:                                        
            saver.restore(sess, checkpoint.model_checkpoint_path)                                  
            print("Successfully  loaded: ", checkpoint.model_checkpoint_path)
            play_dqn(env, parameters_x, readout=readout, s=s)
            print("Thank you for your time !!! Let rate my agent :) !")
        else:
            print("Could not find old network weights")
            exit(-1)


if __name__ == '__main__':
    main()
