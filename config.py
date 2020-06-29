from actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


parameters = {
    'parameters_1':{
        'GAME': 'super_mario_bros-2',
        'NUM_ACTIONS': len(SIMPLE_MOVEMENT),
        'GAMMA': 0.99,
        'OBSERVE': 100000.,
        'EXPLORE': 2000000.,
        'INITIAL_EPSILON': 0.0001,
        'FINAL_EPSILON': 0.0001,
        'REPLAY_MEMORY': 50000,
        'BATCH': 32,
        'FRAME_PER_ACTION': 1
    },

    'parameters_2':{
        'GAME': 'super_mario_bros-2',
        'NUM_ACTIONS': len(SIMPLE_MOVEMENT),
        'GAMMA': 0.99,
        'OBSERVE': 100000.,
        'EXPLORE': 2000000.,
        'INITIAL_EPSILON': 0.001,
        'FINAL_EPSILON': 0.0001,
        'REPLAY_MEMORY': 50000,
        'BATCH': 32,
        'FRAME_PER_ACTION': 1
    },

    'parameters_3':{
        'GAME': 'super_mario_bros-2',
        'NUM_ACTIONS': len(SIMPLE_MOVEMENT),
        'GAMMA': 0.99,
        'OBSERVE': 100000.,
        'EXPLORE': 2000000.,
        'INITIAL_EPSILON': 0.001,
        'FINAL_EPSILON': 0.0001,
        'REPLAY_MEMORY': 50000,
        'BATCH': 64,
        'FRAME_PER_ACTION': 1
    },
}


STANDARD_SIZE = (240, 256)
