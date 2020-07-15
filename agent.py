"""The Implementation of Deep Q Learning for Mario Game"""

from __future__ import print_function

from tqdm import tqdm
import cv2
import sys
import os
import random
import time
import numpy as np
import tensorflow as tf
from collections import deque
from config import *
from actions import ACTION_SPACES
from utils import render_frame
from PIL import Image, ImageDraw


class Agent():

    def __init__(self, env:object, hparameters_x: dict):
        # Get the hyper-parameters
        ###############################################################################
        self.GAME = hparameters_x['GAME']                                             #
        self.NUM_ACTIONS = hparameters_x['NUM_ACTIONS']                               #
        self.GAMMA = hparameters_x['GAMMA']                                           #
        self.OBSERVE = hparameters_x['OBSERVE']                                       #
        self.EXPLORE = hparameters_x['EXPLORE']                                       #
        self.INITIAL_EPSILON = hparameters_x['INITIAL_EPSILON']                       #
        self.FINAL_EPSILON = hparameters_x['FINAL_EPSILON']                           #
        self.REPLAY_MEMORY = hparameters_x['REPLAY_MEMORY']                           #
        self.BATCH = hparameters_x['BATCH']                                           #
        self.FRAME_PER_ACTION = hparameters_x['FRAME_PER_ACTION']                     #
        ###############################################################################

        self.env = env

    def train_dqn(self, readout, s, y, a, sess, train_step, saver, h_fc1, render):
        # store the previous observations in replay memory
        D = deque()
        self.env.reset()

        # Store the logs of training progress to file
        if not os.path.exists("logs_" + self.GAME):
            os.mkdir("logs_" + self.GAME)
        a_file = open("logs_" + self.GAME + "/readout.txt", 'w')
        h_file = open("logs_" + self.GAME + "/hidden.txt", 'w')

        if not os.path.exists("logs_frame"):
            os.mkdir("logs_frame")

        # Get the first state (Frame) by doing nothing and preprocess the image to 80x80x4
        #############################################################################################
        # Chosse the random valid actions(default the action value is discrete: 0, 1,..., 12)       #
        action = self.env.action_space.sample()                                                          #
        # Take the action and get the return information.                                           #
        observation, reward, done, info = self.env.step(action)                                          #
        # Convert the first frame to GRAY scale image with size 80x80                               #
        observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)           #
        # Initiali state is combined from 4 frame -> state is a tensor with 80x80x4 shape           #
        state = np.stack((observation, observation, observation, observation), axis=2)              #
        #############################################################################################

        # start training
        epsilon = self.INITIAL_EPSILON
        t = 0
        done = True

        while True:
            if render:
                self.env.render()

            if done:
                self.env.reset()

            # Choose an action epsilon greedily
            #######################################################################################
            readout_t = readout.eval(feed_dict={s: [state]})[0]                                   #
            action_onehot = np.zeros([self.NUM_ACTIONS])                                               #
                                                                                                  #
            if t % self.FRAME_PER_ACTION == 0:                                                         #
                if random.random() <= epsilon:                                                    #
                    action = random.randrange(self.NUM_ACTIONS)                                        #
                else:                                                                             #
                    action = np.argmax(readout_t)                                                 #
            else:                                                                                 #
                print("Nothing!")                                                                 #
            action_onehot[action] = 1                                                             #
            #######################################################################################

            # Scale down epsilon
            #######################################################################################
            if epsilon > self.FINAL_EPSILON and t > self.OBSERVE:                                           #
                epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE                            #
            #######################################################################################

            # Run the selected action and observe next state and reward
            origin_new_state, reward, done, info = self.env.step(action)
            gray_new_state = cv2.cvtColor(cv2.resize(origin_new_state, (80, 80)), cv2.COLOR_BGR2GRAY)
            new_state = np.reshape(gray_new_state, (80, 80, 1))
            new_state = np.append(new_state, state[:, :, :3], axis=2)

            D.append((state, action_onehot, reward, new_state, done))
            if len(D) > self.REPLAY_MEMORY:
                D.popleft()

            # Training the neural network when when take `observe` steps
            if t >= self.OBSERVE:
                # sample a minibatch to train
                minibatch = random.sample(D, self.BATCH)

                # get the batch variables
                state = [d[0] for d in minibatch]
                action_onehot_batch = [d[1] for d in minibatch]
                reward_bacth = [d[2] for d in minibatch]
                new_state_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_new_state_batch = readout.eval(feed_dict = {s : new_state_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(reward_bacth[i])
                    else:
                        y_batch.append(reward_bacth[i] + GAMMA * np.max(readout_new_state_batch[i]))
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
                saver.save(sess, 'saved_networks/' + self.GAME + '-dqn', global_step = t)

            # print info
            status = ""
            if t <= self.OBSERVE:
                status = "observe"
            elif t > self.OBSERVE and t <= self.OBSERVE + self.EXPLORE:
                status = "explore"
            else:
                status = "train"

            print("TIMESTEP", t, "/ STATE", status, \
                "/ EPSILON", epsilon, "/ ACTION", action, "/ REWARD", reward, \
                "/ Q_MAX %e" % np.max(readout_t))

            # Write logs to file
            if t % 10000 <= 100:
                a_file.write(",".join([str(x) for x in readout_t]) + '\n')
                h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[state]})[0]]) + '\n')
                cv2.imwrite("logs_frame/frame" + str(t) + ".png", gray_new_state)

        self.env.close()

    def train_qlearning(self, render):
        # store the previous observations in replay memory
        size_grid = int(240 / 12)

        done = True
        while True:
            if done:
                self.env.reset()
            #self.env.render()

            action = self.env.action_space.sample()

            observation, done, reward, info = self.env.step(action)

            print(info['stage'])
            input()
            # Devide the windows to 22x22 grids
            observation = observation.astype(np.uint8)


            for i in range(1, 12):
                observation = cv2.line(observation, (0, i * size_grid),
                                    (256, i * size_grid), (0, 0, 0), 1)

            for i in range(1, 12):
                observation = cv2.line(observation, (10 + i * size_grid, 0),
                                    (10 + i * size_grid, 256), (0, 0, 0), 1)

            start_point = (info['x_pos'],  256 - info['y_pos'] + 10)
            end_point = (info['x_pos']+16, 256 - info['y_pos'] + 40)
            observation = cv2.rectangle(observation,
                                        start_point, end_point,
                                        (0, 255, 0), 2)


            cv2.imshow('Draw', observation)
            if cv2.waitKey(0) == 27:
                 break

        cv2.destroyAllWindows()


        # Encoding information
        ######################



        input()
        while True:
            self.env.render()
            if done:
                self.env.reset()
            action = self.env.action_space.sample()
            observation, done, reward, info = self.env.step(action)

        self.env.close()






    def play_dqn(self, readout, s):

        self.env.reset()

        # Get the first state (Frame) by doing nothing and preprocess the image to 80x80x4
        #############################################################################################
        # Chosse the random valid actions(default the action value is discrete: 0, 1,..., 12)       #
        action = self.env.action_space.sample()                                                          #
        # Take the action and get the return information.                                           #
        observation, reward, done, info = self.env.step(action)                                          #
        # Convert the first frame to GRAY scale image with size 80x80                               #
        observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)           #
        # Initiali state is combined from 4 frame -> state is a tensor with 80x80x4 shape           #
        state = np.stack((observation, observation, observation, observation), axis=2)              #
        #############################################################################################
        origin_new_state = state

        t = 0
        done = True

        while True:

            self.env.render()
            #render_frame(frame=origin_new_state, ratio=2)
            #time.sleep(1)

            if done:
                self.env.reset()

            # Choose an action greedily
            #######################################################################################
            readout_t = readout.eval(feed_dict={s: [state]})[0]                                   #
            action_onehot = np.zeros([self.NUM_ACTIONS])                                               #                                                                                             #
            action = np.argmax(readout_t)                                                         #
            action_onehot[action] = 1                                                             #
            #######################################################################################

            # Run the selected action and observe next state and reward
            origin_new_state, reward, done, info = self.env.step(action)

            new_state = cv2.cvtColor(cv2.resize(origin_new_state, (80, 80)), cv2.COLOR_BGR2GRAY)
            new_state = np.reshape(new_state, (80, 80, 1))
            new_state = np.append(new_state, state[:, :, :3], axis=2)

            print("TIMESTEP", t, "/ ACTION", action, \
                "/ REWARD", reward, \
                "/ Q_MAX %e" % np.max(readout_t))


            # Update the new iterator.
            state = new_state
            t += 1

        self.env.close()

if __name__ == '__main__':
    pass
