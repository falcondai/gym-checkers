# A few baseline players for Checkers including a keyboard player

from functools import partial

import numpy as np

from checkers.game import Checkers
from checkers.agents import Player

# A random player
class RandomPlayer(Player):
    '''A player that makes random legal moves.'''
    def next_move(self, board, last_moved_piece):
        state = (board, self.color, last_moved_piece)
        self.simulator.restore_state(state)
        legal_moves = self.simulator.legal_moves()
        move = self.random.choice(np.asarray(legal_moves, dtype='int,int'))
        return move


# Human keyboard player
def keyboard_player_move(board, last_moved_piece):
    '''A player that uses keyboard to select moves.'''
    if last_moved_piece is None:
        input_str = input('* move `from_square, to_square`: ')
    else:
        input_str = input('* move `%i, to_square`: ' % last_moved_piece)
    from_sq, to_sq = map(int, input_str.strip().split(','))
    return from_sq, to_sq

def play_a_game(checkers, black_player, white_player):
    # Play a quick game
    players = {
        'black': black_player,
        'white': white_player,
    }
    ply = 0
    board, turn, last_moved_piece = checkers.save_state()
    winner = None
    while winner is None:
        # The current game state
        checkers.print_board()
        print(ply, turn, last_moved_piece)
        print('legal moves', checkers.legal_moves())
        # Select a legal move for the current player
        from_sq, to_sq = players[turn](board, last_moved_piece)
        board, turn, last_moved_piece, moves, winner = checkers.move(from_sq, to_sq)
        ply += 1
    print(winner, 'wins')
    return winner


if __name__ == '__main__':
    ch = Checkers()
    ch.print_empty_board()

    black_random_player = RandomPlayer('black', seed=0)
    white_random_player = RandomPlayer('white', seed=1)
    play_a_game(ch, black_random_player.next_move, white_random_player.next_move)
    # play_a_game(ch, keyboard_player_move, keyboard_player_move)
