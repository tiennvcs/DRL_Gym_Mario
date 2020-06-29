"""The Implementation of Deep Q Learning for Mario Game"""

from __future__ import print_function

from tqdm import tqdm
import cv2
import sys
import random
import numpy as np
from collections import deque
from config import *
from deep_q_network import createNetwork
from actions import ACTION_SPACES
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def play_dqn(env, parameters_x):
    """
    Play the environment making uniformly random decisions.

    Args:
        env (gym.Env): the initialized gym environment to play
        steps (int): the number of random steps to take

    Returns:
        None

    """
    # Get the hyper-parameters
    GAME = parameters_x['GAME']
    NUM_ACTIONS = parameters_x['NUM_ACTIONS']
    GAMMA = parameters_x['GAMMA']
    OBSERVE = parameters_x['OBSERVE']
    EXPLORE = parameters_x['EXPLORE']
    INITIAL_EPSILON = parameters_x['INITIAL_EPSILON']
    FINAL_EPSILON = parameters_x['FINAL_EPSILON']
    REPLAY_MEMORY = parameters_x['REPLAY_MEMORY']
    BATCH = parameters_x['BATCH']
    FRAME_PER_ACTION = parameters_x['FRAME_PER_ACTION']

    sess = tf.InteractiveSession()
    # Initialize a neural network
    s, readout, h_fc1 = createNetwork(parameters_x)

    # define the cost function
    a = tf.placeholder("float", [None, NUM_ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # store the previous observations in replay memory
    D = deque()
    env.reset()

    # Get the first state (Frame) by doing nothing and preprocess the image to 80x80x4
    #############################################################################################
    # Chosse the random valid actions(default the action value is discrete: 0, 1,..., 12)       #
    action = env.action_space.sample()
    # Take the action and get the return information.                                           #
    observation, reward, done, info = env.step(action)                                          #
    # Convert the first frame to GRAY scale image with size 80x80                               #
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)           #
    # Initiali state is combined from 4 frame -> state is a tensor with 80x80x4 shape           #
    state = np.stack((observation, observation, observation, observation), axis=2)              #
    #############################################################################################

    # Saving and loading networks
    ############################################################################################
    saver = tf.train.Saver()                                                                   #
    sess.run(tf.initialize_all_variables())                                                    #
    checkpoint = tf.train.get_checkpoint_state("saved_networks")                               #
    if checkpoint and checkpoint.model_checkpoint_path:                                        #
        saver.restore(sess, checkpoint.model_checkpoint_path)                                  #
        print("Successfully  loaded: ", checkpoint.model_checkpoint_path)                      #
    else:                                                                                      #
        print("Could not find old network weights")                                            #
    ############################################################################################

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    done = True

    while True:
        if done:
            env.reset()

        # Choose an action epsilon greedily
        #######################################################################################
        readout_t = readout.eval(feed_dict={s: [state]})[0]                                   #
        action_onehot = np.zeros([NUM_ACTIONS])                                               #
                                                                                              #
        if t % FRAME_PER_ACTION == 0:                                                         #
            if random.random() <= epsilon:                                                    #
                action = random.randrange(NUM_ACTIONS)                                        #
            else:                                                                             #
                action = np.argmax(readout_t)                                                 #
        else:                                                                                 #
            print("Nothing!")                                                                 #
        action_onehot[action] = 1                                                             #
        #######################################################################################

        # Scale down epsilon
        #######################################################################################
        if epsilon > FINAL_EPSILON and t > OBSERVE:                                           #
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE                            #
        #######################################################################################

        # Run the selected action and observe next state and reward
        new_state, reward, done, info = env.step(action)

        new_state = cv2.cvtColor(cv2.resize(new_state, (80, 80)), cv2.COLOR_BGR2GRAY)
        new_state = np.reshape(new_state, (80, 80, 1))
        new_state = np.append(new_state, state[:, :, :3], axis=2)

        D.append((state, action_onehot, reward, new_state, done))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # Training the neural network when when take `observe` steps
        if t >= OBSERVE:
            # sample a minibatch to train
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            state = [d[0] for d in minibatch]
            action_onehot_batch = [d[1] for d in minibatch]
            reward_bacth = [d[2] for d in minibatch]
            new_state_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : new_state_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(reward_bacth[i])
                else:
                    y_batch.append(reward_bacth[i] + GAMMA * np.max(new_state_batch[i]))
            #print(y_batch)
            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : action_onehot_batch,
                s : new_state_batch}
            )

        # Update the new iterator.
        state = new_state
        t += 1

        # Save the progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info

        status = ""
        if t <= OBSERVE:
            status = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            status = "explore"
        else:
            status = "train"

        print("TIMESTEP", t, "/ STATE", status, \
            "/ EPSILON", epsilon, "/ ACTION", action, "/ REWARD", reward, \
            "/ Q_MAX %e" % np.max(readout_t))

        env.render()

    env.close()


# explicitly define the outward facing API of this module
__all__ = [play_dqn.__name__]
