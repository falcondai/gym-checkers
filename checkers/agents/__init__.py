import numpy as np

from checkers.game import Checkers


class Player:
    '''An abstract player.'''
    def __init__(self, color, seed=None):
        self.color = color
        self.simulator = Checkers()
        self.random = np.random.RandomState(seed=seed)

    def next_move(self, board, last_moved_piece):
        raise NotImplementedError
