# Minimax with alpha-beta pruning and a hand-crafted valuation function

import time

import numpy as np

from checkers.game import Checkers
from checkers.agents import Player

class MinimaxPlayer(Player):
    '''Minimax search with alpha-beta pruning'''
    # The value of all the outcomes
    win, draw, loss = 1, 0, -1
    def __init__(self, color, value_func=None, terminate_rollout_func=None, rollout_order_gen=None, seed=None):
        super().__init__(color=color, seed=seed)
        self.adversary = 'black' if self.color == 'white' else 'white'
        self.value = value_func
        # Default to rollout till the end
        self.should_stop_rollout = terminate_rollout_func or (lambda state, depth : False)
        # Default to evaluate actions at a random order
        self.rollout_order = rollout_order_gen or (lambda moves : self.random.permutation(np.asarray(moves, dtype='int,int')))
        # Cache the evaluated values
        self.cached_values = {}

    @staticmethod
    def immutable_state(board, turn, last_moved_piece):
        pieces = []
        for player in Checkers.all_players:
            for piece_type in Checkers.all_piece_types:
                pieces.append(frozenset(board[player][piece_type]))
        return tuple(pieces), turn, last_moved_piece

    def next_move(self, board, last_moved_piece):
        state = board, self.color, last_moved_piece
        t0 = time.time()
        max_value, max_move, n_moves = self.max_step(state, 0, set())
        dt = time.time() - t0
        print('evaluated %i positions in %.2fs (avg %.2f positions/s)' % (n_moves, dt, n_moves / dt))
        return max_move

    def max_step(self, state, depth, visited_states):
        assert state[1] == self.color, 'Max step should be executed in the player\'s turn.'

        im_state = MinimaxPlayer.immutable_state(*state)
        # Already evaluated?
        if im_state in self.cached_values:
            return self.cached_values[im_state], None, 0

        # Evaluate this state
        self.simulator.restore_state(state)
        moves = self.simulator.legal_moves()
        # Base case. Win/loss check
        if len(moves) == 0:
            # No available moves => loss
            print('max', depth, 'end', -1)
            self.cached_values[im_state] = MinimaxPlayer.loss
            return MinimaxPlayer.loss, None, 1
        # Loop checking for draws
        if im_state in visited_states:
            print('max', depth, 'end', 0)
            self.cached_values[im_state] = MinimaxPlayer.draw
            return MinimaxPlayer.draw, None, 1
        else:
            visited_states.add(im_state)
        # Should we terminate the rollout
        if self.should_stop_rollout(state, depth):
            return self.value(state), None, 1
        # Rollout each legal move
        max_value, max_move, evaluated_moves = None, None, 0
        for move in self.rollout_order(moves):
            print('max', depth, move, state[0])
            self.simulator.restore_state(state)
            self.simulator.print_board()
            next_board, next_turn, next_last_moved_piece, next_moves, winner = self.simulator.move(*move, skip_check=True)
            next_state = self.simulator.save_state()
            if next_turn == self.color:
                # Still our turn
                value, _, n_moves = self.max_step(next_state, depth=depth + 1, visited_states=visited_states)
            else:
                # Our adversary's turn
                value, _, n_moves = self.min_step(next_state, depth=depth + 1, visited_states=visited_states)
            # Update the max_value
            if max_value is None or max_value < value:
                max_value = value
                max_move = move
            # Update statistics
            evaluated_moves += n_moves
        return max_value, max_move, evaluated_moves

    def min_step(self, state, depth, visited_states):
        assert state[1] == self.adversary, 'Min step should be executed in an adversary\'s turn.'

        im_state = MinimaxPlayer.immutable_state(*state)
        # Already evaluated?
        if im_state in self.cached_values:
            return self.cached_values[im_state], None, 0

        # Evaluate this state
        self.simulator.restore_state(state)
        moves = self.simulator.legal_moves()
        # Base case. Win/loss check
        if len(moves) == 0:
            # No available moves for adversary => win
            print('min', depth, 'end', 1)
            self.cached_values[im_state] = MinimaxPlayer.win
            return MinimaxPlayer.win, None, 1
        # Loop checking for draws
        if im_state in visited_states:
            print('min', depth, 'end', 0)
            self.cached_values[im_state] = MinimaxPlayer.draw
            return MinimaxPlayer.draw, None, 1
        else:
            visited_states.add(im_state)
        # Should we terminate the rollout
        if self.should_stop_rollout(state, depth):
            return self.value(state), None, 1
        # Rollout each legal move
        min_value, min_move, evaluated_moves = None, None, 0
        for move in self.rollout_order(moves):
            print('min', depth, move)
            self.simulator.restore_state(state)
            next_board, next_turn, next_last_moved_piece, next_moves, winner = self.simulator.move(*move, skip_check=True)
            next_state = self.simulator.save_state()
            if next_turn == self.color:
                # Still our turn
                value, _, n_moves = self.max_step(next_state, depth=depth + 1, visited_states=visited_states)
            else:
                # Our adversary's turn
                value, _, n_moves = self.min_step(next_state, depth=depth + 1, visited_states=visited_states)
            # Update the min_value
            if min_value is None or value < min_value:
                min_value = value
                min_move = move
            # Update statistics
            evaluated_moves += n_moves
        return min_value, min_move, evaluated_moves


if __name__ == '__main__':
    from checkers.agents.baselines import play_a_game, RandomPlayer
    board = Checkers.empty_board()
    board['black']['men'].update([7, 14])
    board['white']['men'].update([17, 11])
    ch = Checkers(board=board, turn='white')
    ch.print_board()
    player = MinimaxPlayer('white')
    move = player.next_move(ch.board, ch.last_moved_piece)
    print(move)
    print(ch.move(*move, skip_check=True))

    # ch = Checkers()
    # black_player = MinimaxPlayer('black')
    # white_player = RandomPlayer('white')
    # play_a_game(ch, black_player.next_move, white_player.next_move)
