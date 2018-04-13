# Monte Carlo tree search
from __future__ import absolute_import, division, print_function
from six.moves import range


import time, copy, itertools
from functools import partial
from collections import defaultdict

import numpy as np

from checkers.game import Checkers
from checkers.agents import Player


class MctsPlayer(Player):
    '''
    Monte Carlo Tree Search player with upper confidence bound (UCB1)
    reference: https://www.cs.swarthmore.edu/~bryce/cs63/s16/slides/2-17_extending_mcts.pdf
    '''
    def __init__(self, color, exploration_coeff=1, max_rounds=800, max_plies=float('inf'), seed=None):
        super(MctsPlayer, self).__init__(color=color, seed=seed)

        # Default policy for rollouts
        self.rollout_policy = lambda moves: self.random.choice(np.asarray(moves, dtype='int,int'))
        # Maximum plies in each rollout
        self.max_plies = max_plies
        # Maximum rounds of simulation
        self.max_rounds = max_rounds
        # Explore
        self.exploration_coeff = exploration_coeff

        # Checkers can draw. Should keep the win counts separately for each player in general
        self.stats = defaultdict(lambda: (0, 0, 0))
        self.children = defaultdict(lambda : set())

    def q(self, turn, st):
        # XXX compute Q for each player. Both try to maximize its own value
        black_wins, white_wins, n_samples = self.stats[st]
        next_q = black_wins / n_samples if turn == 'black' else white_wins / n_samples
        return next_q

    @staticmethod
    def successor(st):
        sim = Checkers()
        state = MctsPlayer.convert_to_state(st)
        sim.restore_state(state)
        next_sts = []
        moves = sorted(sim.legal_moves())
        for move in moves:
            sim.restore_state(state)
            board, turn, last_moved_piece, _, winner = sim.move(*move)
            next_state = board, turn, last_moved_piece
            next_st = MctsPlayer.immutable_state(*next_state)
            next_sts.append(next_st)
        return next_sts

    def next_move(self, board, last_moved_piece):
        # Initialize with root node
        st0 = MctsPlayer.immutable_state(board, self.color, last_moved_piece)

        round = 0
        while round < self.max_rounds:
            # Start from the root
            st = st0
            walked_sts = [st]
            # Follow Q for non-leaf nodes
            while 0 < len(self.children[st]):
                # In-tree, choose a successor according to statistics
                if self.random.rand() < 0.1:
                    # Explore randomly
                    break
                else:
                    # Select a visited successor state according to Q
                    next_sts = list(self.children[st])
                    max_q = float('-inf')
                    next_idx = self.random.randint(len(next_sts))
                    max_st = next_sts[next_idx]
                    turn = st[1]
                    for next_st in next_sts:
                        # Upper confidence bound
                        next_q = self.q(turn, next_st) + self.exploration_coeff * MctsPlayer.ucb(self.stats[st][-1], self.stats[next_st][-1])
                        if max_q < next_q:
                            max_q = next_q
                            max_st = next_st
                    st = max_st
                # Add it to walked states in this round
                walked_sts.append(st)
            # Out-of-tree, choose a successor state randomly
            next_sts = MctsPlayer.successor(st)
            if 0 < len(next_sts):
                next_idx = self.random.randint(len(next_sts))
                next_st = next_sts[next_idx]
                walked_sts.append(next_st)
                # Add this node to the tree
                self.children[st].add(next_st)
                st = next_st
            # Rollout till the game ends
            winner = self.rollout(st)
            # Update statistics on walked nodes
            for st in walked_sts:
                black_wins, white_wins, n_samples = self.stats[st]
                turn = st[1]
                # Update wins based on the turn
                black_wins += 1 if winner == 'black' else 0
                white_wins += 1 if winner == 'white' else 0
                self.stats[st] = black_wins, white_wins, n_samples + 1
            round += 1

        # Select a move after searching
        print(len(self.children[st0]))
        sim = Checkers()
        state = MctsPlayer.convert_to_state(st0)
        sim.restore_state(state)
        moves = sorted(sim.legal_moves())
        max_q, max_q_move = float('-inf'), None
        max_n, max_n_move = float('-inf'), None
        for move in moves:
            sim.restore_state(state)
            board, turn, last_moved_piece, _, _ = sim.move(*move)
            next_st = MctsPlayer.immutable_state(board, turn, last_moved_piece)
            if next_st in self.children[st0]:
                # Maximize Q for the player
                next_q = self.q(self.color, next_st)
                if max_q < next_q:
                    max_q = next_q
                    max_q_move = move
                n_samples = self.stats[next_st][-1]
                if max_n < n_samples:
                    max_n = n_samples
                    max_n_move = move
                print(move, '%.2f' % next_q, n_samples, next_st[1], self.stats[next_st], self.stats[next_st][0] + self.stats[next_st][1] - self.stats[next_st][-1])
        print('%.2f' % self.q(self.color, st0))
        print('leaves depth histogram (depth, count):', sorted(MctsPlayer.hist_leaf_depth(self.children, st0).items()))
        return max_q_move

    @staticmethod
    def immutable_state(board, turn, last_moved_piece):
        return Checkers.immutable_board(board), turn, last_moved_piece

    @staticmethod
    def hist_leaf_depth(children, root_st):
        '''Returns counts of leaves at different depths'''
        counts = defaultdict(lambda : 0)
        # Breadth first iteration
        queue = [(root_st, 0)]
        # XXX be careful with loops in a graph
        visited = set()
        while 0 < len(queue):
            st, depth = queue.pop(0)
            if st in visited:
                # Found a loop
                continue
            visited.add(st)
            if 0 == len(children[st]):
                # A leaf node
                counts[depth] += 1
            else:
                # Internal node, expand
                queue += itertools.product(children[st], [depth + 1])
        return counts

    @staticmethod
    def convert_to_state(st):
        bo, turn, last_moved_piece = st
        black_men, black_kings, white_men, white_kings = bo
        board = {
            'black': {
                'men': set(black_men),
                'kings': set(black_kings),
                },
            'white': {
                'men': set(white_men),
                'kings': set(white_kings),
            },
        }
        return board, turn, last_moved_piece

    def rollout(self, st):
        '''Rollout till the game ends with a win/draw'''
        sim = Checkers()
        state = MctsPlayer.convert_to_state(st)
        sim.restore_state(state)
        ply = 0
        moves = sim.legal_moves()
        # Terminal state
        if len(moves) == 0:
            winner = 'white' if st[1] == 'black' else 'black'
        else:
            winner = None
        while ply < self.max_plies and winner is None:
            from_sq, to_sq = self.rollout_policy(sorted(moves))
            board, turn, last_moved_piece, moves, winner = sim.move(from_sq, to_sq, skip_check=True)
            ply += 1
        # Returns the winner or None in a draw
        return winner

    @staticmethod
    def ucb(n_parent_visits, n_visits):
        '''Upper confidence bound (UCB1) based on Hoeffding inequality'''
        return np.sqrt(2 * np.log(n_parent_visits) / n_visits)

if __name__ == '__main__':
    from checkers.agents.baselines import play_a_game, RandomPlayer
    from checkers.agents.alpha_beta import MinimaxPlayer
    ch = Checkers()
    # black_player = RandomPlayer('black')
    black_player = MinimaxPlayer(
        color='black',
        # The provided legal moves might be ordered differently
        rollout_order_gen=lambda x : sorted(x),
        search_depth=3,
        seed=0,
        )
    white_player = MctsPlayer(
        color='white',
        exploration_coeff=1,
        max_rounds=400,
        seed=1,
        )
    play_a_game(ch, black_player.next_move, white_player.next_move)
