import numpy as np

from checkers.game import Checkers


class Player:
    '''An abstract player.'''
    def __init__(self, color, seed=None):
        # Which side is being played
        self.color = color
        # Internal simulator for rollouts
        self.simulator = Checkers()
        # Fixing the random state for easy replications
        self.random = np.random.RandomState(seed=seed)

    def next_move(self, board, last_moved_piece):
        raise NotImplementedError
