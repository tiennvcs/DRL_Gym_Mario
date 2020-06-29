# Deep Reinforcement Learning in Super Mario Bros
----

Super Mario Bros game is the classical video game and we play game by interact with the physical buttons.
Instead play the game by hand, we apply one of the most powerful technique of Deep Reinforcement Learning (DRL), that is Deep Q Network - the combination of Convolution Neuron Network with Q learning algorithm.

## Description about project
- Implement the Deep Q Network for Gym Sumper Mario Bros game.
- <strike>Eperiment some Neuron network architectures appied for problem.</strike>
- *The source code use Python 3.7.7, gym-super-mario-bros 7.3.2, Tensorflow 1.15.2, OpenCV 4.2.0.34.*

## To run projects
  ***Clone the repository to your local machine***
  ```bash
    $ git clone https://github.com/tiennvuit/DRL_Gym_Mario.git
  ```
  
  ### Setting the environment
  #### With Anaconda platform
  ```bash
    $ conda create --name drl_mario_env python==3.7.*
    
    $ conda activate drl_mario_env
    
    (drl_mario_env)$ pip install -r requirements.txt
  ```
  
  #### With Docker (*Only recommomend for training period*)**
  ```bash
    $ sudo docker container run -it -v /directory/of/repository:/home --name drl_mario tensorflow/tensorflow:1.15.2
    
    $ apt-get -y update && apt-get -y libsm6 libxext6 libxrender-dev && pip install --upgrade pip
    
    $ pip install gym_super_mario_bros && pip install opencv-python
 
  ```
 
 #### With virtual enviroment
 ```bash
  $ python -m venv env
  
  $ source env/bin/activate
  
  (env) $ pip install --upgrade pip setuptools
  
  (env) $ pip install -r requirements.txt
 ```
 
  **Training an agent to play Mario game with default Hyper-parameters**.
  ```bash
    python main.py -m train
  ```
  **See the agent play the game with the model learnt from the training**:
  ```bash
    python main.py -m play
  ```
  
  
## Experientials
- We train the agent with default hyper-parameters:
    + 
 

## References:
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [DeepLearningFlappyBird](https://github.com/tiennvuit/DeepLearningFlappyBird)
