import numpy as np
import pandas as pd
from copy import copy


class PayoffMatrix:

    def __init__(self, payoffs=None, p1_strategies=None, p2_strategies=None, file_source=None):
        if file_source is None:
            # initialize payoff matrix as a numpy array
            self.payoffs = np.array(payoffs, dtype=np.float)
            # if strategy names are not set or have the wrong length, give them index labels
            if p1_strategies is None or len(p1_strategies) != self.payoffs.shape[0]:
                self.p1_strategies = np.arange(self.payoffs.shape[0])
            else:
                self.p1_strategies = copy(p1_strategies)
            if p2_strategies is None or len(p2_strategies) != self.payoffs.shape[1]:
                self.p1_strategies = np.arange(self.payoffs.shape[1])
            else:
                self.p2_strategies = copy(p2_strategies)
        else:
            # convert 2d string array to 3d array of floats
            df = pd.read_csv(file_source, index_col=0, header=0)
            self.payoffs = np.zeros((df.values.shape[0], df.values.shape[1], 2))
            for i in range(df.values.shape[0]):
                for j in range(df.values.shape[1]):
                    for k, p in enumerate(df.values[i, j].split(',')):
                        self.payoffs[i, j, k] = float(p)
            # read index/column names from dataframe and use them as player 1 / player 2 strategy names
            self.p1_strategies = list(df.index.values)
            self.p2_strategies = list(df.columns.values)
        # if possible, convert payoffs to integers
        if np.all(self.payoffs % 1 == 0):
            self.payoffs = self.payoffs.astype(np.int)

    def best_responses(self, player, opp_strategy):
        if player == 1:
            strategy_names = self.p1_strategies
            # get column number of opponent's strategy and the corresponding payoffs
            column_number = self.p2_strategies.index(opp_strategy)
            payoffs = self.payoffs[:, column_number, 0]
        else:
            strategy_names = self.p2_strategies
            # get row number of opponent's strategy and the corresponding payoffs
            row_number = self.p1_strategies.index(opp_strategy)
            payoffs = self.payoffs[row_number, :, 1]
        # get value of maximum payoff given that the opponent plays opp_strategy
        best_payoff = np.max(payoffs)
        # get indices of player's strategies that yield best_payoff
        best_indices = np.argwhere(payoffs == best_payoff).flatten()
        # return corresponding strategy names
        return [strategy_names[i] for i in best_indices]

    def dominated_strategies(self, player, mode='strict'):
        dominated_strategies = {}
        # iterate over all of player's strategies
        for n_strategy in range(self.payoffs.shape[player - 1]):
            # get payoffs of current strategy across all strategies of opponent
            payoffs = self.payoffs[n_strategy, :, 0] if player == 1 else self.payoffs[:, n_strategy, 1]
            # iterate over all other player's strategies
            for n_other_strategy in range(self.payoffs.shape[player - 1]):
                if n_other_strategy == n_strategy:
                    continue
                # get payoffs of current other strategy across all strategies of opponent
                if player == 1:
                    other_payoffs = self.payoffs[n_other_strategy, :, 0]
                else:
                    other_payoffs = self.payoffs[:, n_other_strategy, 1]
                # add strategy to dominated strategies if it is strictly/weakly dominated by the other strategy
                strictly_dominated = np.all(payoffs < other_payoffs)
                weakly_dominated = not strictly_dominated and np.all(payoffs <= other_payoffs) \
                    and np.any(payoffs < other_payoffs)
                if mode == 'strict' and strictly_dominated or mode == 'weak' and weakly_dominated:
                    strategy_names = self.p1_strategies if player == 1 else self.p2_strategies
                    dominated_strategies[strategy_names[n_strategy]] = strategy_names[n_other_strategy]
                    break
        return dominated_strategies

    def eliminate_strategy(self, player, strategy):
        if player == 1:
            # get row number of strategy, then remove it from list
            index = self.p1_strategies.index(strategy)
            self.p1_strategies.remove(strategy)
        else:
            # get column number of strategy, then remove it from list
            index = self.p2_strategies.index(strategy)
            self.p2_strategies.remove(strategy)
        # remove corresponding row/column from payoff matrix
        self.payoffs = np.delete(self.payoffs, index, player-1)

    def output(self, target_file=None):
        # convert payoff matrix to a 2d string array 
        rows, columns = self.payoffs.shape[:2]
        payoffs_2d = [['{}, {}'.format(*self.payoffs[i, j]) for j in range(columns)] for i in range(rows)]
        # create pandas dataframe, convert it to a string, and print it
        payoffs_df = pd.DataFrame(payoffs_2d, index=self.p1_strategies, columns=self.p2_strategies)
        print(payoffs_df.to_string())
        # export dataframe to a csv file (if given)
        if target_file is not None:
            payoffs_df.to_csv(target_file)
