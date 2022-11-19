# Guide Mario playing in 2D environment using Deep reinforcement learning

Apply DRL to design an algorithm that can teach agents to automatically play control agent-based games. The challenge is how to guide the agent to act in a 2-dimensional environment to satisfy the maximum score. Utilizing Q-learning as the objective function and Neural network as a feature extractor to guide Mario to learn a policy that maximizes the total reward in a single run. We experiment with two algorithms: Deep Q learning and Deep Double Q learning on the classical control-agent game: Super Mario with different complex scenes and report the achieving maximum distance.

## Usages
  **Training an agent to play Mario game with default Hyper-parameters**.
  ```bash
    python main.py -m train -p 1 -r 0          # Training agent with the 1'st hyper-parameter and not render the graphic.
    
    
  ```
  **See the agent play the game with the model learnt from the training**:
  ```bash
    python main.py -m play
  ```
  
## Setting the environment
  ### With Anaconda platform
  ```bash
    $ conda create --name drl_mario_env python==3.7.*
    
    $ conda activate drl_mario_env
    
    $ pip install -r requirements.txt
  ```
  
  ### With Docker (*Only recommomend for training period*)**
  ```bash
    $ sudo docker container run -it -v /directory/of/repository:/home--name drl_mario ubuntu:18.04
    
    $ apt-get update
    
    $ apt-get install python3-pip
    
    $ pip install --upgrade pip setuptools
    
    $ pip install -r requirements.txt
  ```
 
 ### With virtual enviroment
 ```bash
  $ python -m venv env
  
  $ source env/bin/activate
  
  (env) $ pip install --upgrade pip setuptools
  
  (env) $ pip install -r requirements.txt
 ```
 

  
 

## References:
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [DeepLearningFlappyBird](https://github.com/tiennvuit/DeepLearningFlappyBird)
