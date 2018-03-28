# Minimax with alpha-beta pruning and a hand-crafted valuation function

import time, math
from functools import partial

import numpy as np

from checkers.game import Checkers
from checkers.agents import Player

class MinimaxPlayer(Player):
    '''Minimax search with alpha-beta pruning'''
    # The value of all the outcomes
    win, draw, loss = 1, 0, -1
    def __init__(self, color, value_func=None, search_depth=math.inf, rollout_order_gen=None, seed=None):
        super().__init__(color=color, seed=seed)
        self.adversary = 'black' if self.color == 'white' else 'white'
        # Default to evaluate using piece values
        self.value = value_func or partial(piece_count_heuristic, self.color, 1, 1)
        # Default to evaluate actions at a random order
        self.rollout_order = rollout_order_gen or (lambda moves : self.random.permutation(np.asarray(moves, dtype='int,int')))
        # Cache the evaluated values
        self.cached_values = {}
        self.search_depth = search_depth
        self.n_evaluated_positions = 0
        self.evaluation_dt = 0

    @staticmethod
    def immutable_state(board, turn, last_moved_piece):
        pieces = []
        for player in Checkers.all_players:
            for piece_type in Checkers.all_piece_types:
                pieces.append(frozenset(board[player][piece_type]))
        return tuple(pieces), turn, last_moved_piece

    def add_to_cache(self, immutable_state, value):
        # TODO evict some cache to prevent over-capacity
        self.cached_values[immutable_state] = value

    def next_move(self, board, last_moved_piece):
        state = board, self.color, last_moved_piece
        t0 = time.time()
        self.simulator.restore_state(state)
        moves = self.simulator.legal_moves()
        if len(moves) == 1:
            # No other choice
            best_move = moves[0]
        else:
            # More than one legal move
            max_value, best_move = -math.inf, None
            for move in moves:
                self.simulator.restore_state(state)
                self.simulator.move(*move, skip_check=True)
                value = self.minimax_search(state, -math.inf, +math.inf, self.search_depth, set())
                if max_value < value:
                    max_value = value
                    best_move = move
        dt = time.time() - t0
        self.evaluation_dt += dt
        return best_move

    def minimax_search(self, state, alpha, beta, depth, visited_states):
        # TODO use a stack for depth-first search?
        # XXX visited_states should only track a path?
        '''Bounded depth first minimax search with alpha-beta pruning'''
        board, turn, last_moved_piece = state
        im_state = MinimaxPlayer.immutable_state(*state)

        # Already evaluated?
        if im_state in self.cached_values:
            # print('cache hit')
            self.cached_values[im_state]

        # Evaluate this state
        self.simulator.restore_state(state)
        moves = self.simulator.legal_moves()
        # Base case. Win/loss check
        if len(moves) == 0:
            # No available moves => loss
            value = MinimaxPlayer.loss if turn == self.color else MinimaxPlayer.win
            self.add_to_cache(im_state, value)
            # print(self.color == turn, depth, 'end', value)
            self.n_evaluated_positions += 1
            return value
        # # Loop checking for draws
        # if im_state in visited_states:
        #     # print(self.color == turn, depth, 'end', 0)
        #     value = MinimaxPlayer.draw
        #     self.add_to_cache(im_state, value)
        #     self.n_evaluated_positions += 1
        #     return value
        # else:
        #     visited_states.add(im_state)

        # We should terminate the rollout early
        if depth == 0:
            # Terminate with a valuation function
            value = self.value(*state)
            # print(self.color == turn, depth, 'end', value)
            self.n_evaluated_positions += 1
            return value
        # Rollout each legal move
        if turn == self.color:
            # Maximizing node
            extreme_value = alpha
            for move in self.rollout_order(moves):
                # print(self.color == turn, depth, move, state[0])
                self.simulator.restore_state(state)
                # self.simulator.print_board()
                next_board, next_turn, next_last_moved_piece, next_moves, winner = self.simulator.move(*move, skip_check=True)
                next_state = self.simulator.save_state()
                # Evaluate the next position
                value = self.minimax_search(next_state, extreme_value, beta, depth=depth-1, visited_states=visited_states)
                # Update the max value
                extreme_value = max(value, extreme_value)
                if beta < extreme_value:
                    # Prune the rest of children nodes
                    return beta
        else:
            # Minimizing node
            extreme_value = beta
            for move in self.rollout_order(moves):
                # print(self.color == turn, depth, move, state[0])
                self.simulator.restore_state(state)
                # self.simulator.print_board()
                next_board, next_turn, next_last_moved_piece, next_moves, winner = self.simulator.move(*move, skip_check=True)
                next_state = self.simulator.save_state()
                # Evaluate the next position
                value = self.minimax_search(next_state, alpha, extreme_value, depth=depth-1, visited_states=visited_states)
                # Update the min value
                extreme_value = min(value, extreme_value)
                if extreme_value < alpha:
                    # Prune the rest of children nodes
                    return alpha
        return extreme_value


def piece_count_heuristic(color, king_weight, man_weight, board, turn, last_moved_piece):
    '''Advantage in the total piece value'''
    n_black_pieces = man_weight * len(board['black']['men']) + king_weight * len(board['black']['kings'])
    n_white_pieces = man_weight * len(board['white']['men']) + king_weight * len(board['white']['kings'])
    return (1 if color == 'black' else -1) * (n_black_pieces - n_white_pieces) / max(king_weight, man_weight) / 12


if __name__ == '__main__':
    from checkers.agents.baselines import play_a_game, RandomPlayer, keyboard_player_move

    # board = Checkers.empty_board()
    # board['black']['men'].update([7, 14])
    # board['white']['men'].update([17, 11])
    # ch = Checkers(board=board, turn='white')
    # ch.print_board()

    # player = MinimaxPlayer('white', value_func=partial(piece_count_heuristic, 'white'), depth=1, seed=0)
    # move = player.next_move(ch.board, ch.last_moved_piece)
    # print(move)
    # print(ch.move(*move, skip_check=True))

    ch = Checkers()

    black_player = MinimaxPlayer('black', value_func=partial(piece_count_heuristic, 'black', 2, 1), search_depth=4)
    # white_player = MinimaxPlayer('white', value_func=partial(piece_count_heuristic, 'white', 2, 1), search_depth=2)
    white_player = RandomPlayer('white')
    play_a_game(ch, black_player.next_move, white_player.next_move)
    # play_a_game(ch, keyboard_player_move, white_player.next_move)
    print('black player evaluated %i positions in %.2fs (avg %.2f positions/s)' % (black_player.n_evaluated_positions, black_player.evaluation_dt, black_player.n_evaluated_positions / black_player.evaluation_dt))
    # print('white player evaluated %i positions in %.2fs (avg %.2f positions/s)' % (white_player.n_evaluated_positions, white_player.evaluation_dt, white_player.n_evaluated_positions / white_player.evaluation_dt))
