import numpy as np
import pandas as pd


def __show_best_responses(payoff_matrix, is_best_response):
    # convert payoff matrix to a 2d string array with best responses marked with *
    payoffs_2d = []
    for i in range(payoff_matrix.payoffs.shape[0]):
        row = []
        for j in range(payoff_matrix.payoffs.shape[1]):
            entry = '{}{},'.format(payoff_matrix.payoffs[i, j, 0], '*' if is_best_response[i, j, 0] else '')
            entry += ' {}{}'.format(payoff_matrix.payoffs[i, j, 1], '*' if is_best_response[i, j, 1] else '')
            row.append(entry)
        payoffs_2d.append(row)
    # create pandas dataframe, convert it to a string, and print it
    payoffs_df = pd.DataFrame(payoffs_2d, index=payoff_matrix.p1_strategies, columns=payoff_matrix.p2_strategies)
    print(payoffs_df.to_string() + '\n\n')


def best_responses(payoff_matrix, show_steps=True):
    # initialize matrix of zeros to indicate whether a response is a best response
    is_best_response = np.zeros((payoff_matrix.payoffs.shape[0], payoff_matrix.payoffs.shape[1], 2))
    # iterate over all of player 2's strategies
    for j, p2_strategy in enumerate(payoff_matrix.p2_strategies):
        # get player 1's best responses to p2_strategy
        p1_best_responses = payoff_matrix.best_responses(player=1, opp_strategy=p2_strategy)
        # set the corresponding entries in is_best_response to 1
        for p1_best_response in p1_best_responses:
            row_number = payoff_matrix.p1_strategies.index(p1_best_response)
            is_best_response[row_number, j, 0] = 1
        # print result and visualize best responses in the payoff matrix
        if show_steps:
            output = ', '.join([str(s) for s in p1_best_responses])
            if len(p1_best_responses) > 1:
                output += " are player 1's best responses to player 2 playing {}.\n".format(p2_strategy)
            else:
                 output += " is player 1's best response to player 2 playing {}.\n".format(p2_strategy)
            print(output)
            __show_best_responses(payoff_matrix, is_best_response)
    # iterate over all of player 1's strategies
    for i, p1_strategy in enumerate(payoff_matrix.p1_strategies):
        # get player 2's best responses to p1_strategy
        p2_best_responses = payoff_matrix.best_responses(player=2, opp_strategy=p1_strategy)
        # set the corresponding entries in is_best_response to 1
        for p2_best_response in p2_best_responses:
            column_number = payoff_matrix.p2_strategies.index(p2_best_response)
            is_best_response[i, column_number, 1] = 1
        # print result and visualize best responses in the payoff matrix
        if show_steps:
            output = ', '.join([str(s) for s in p2_best_responses])
            if len(p2_best_responses) > 1:
                output += " are player 2's best responses to player 1 playing {}.\n".format(p1_strategy)
            else:
                 output += " is player 2's best response to player 1 playing {}.\n".format(p1_strategy)
            print(output)
            __show_best_responses(payoff_matrix, is_best_response)
    # get indices of nash equilibria (best response of both players) and the corresponding strategy names
    psne_indices = np.argwhere(np.sum(is_best_response, axis=2) == 2)
    psne = [(payoff_matrix.p1_strategies[i], payoff_matrix.p2_strategies[j]) for i, j in psne_indices]
    # print final result and return the nash equilibria
    if len(psne) == 0 and show_steps:
        print('There are no pure strategy Nash equilibria.')
    elif len(psne) == 1 and show_steps:
        print('<{}, {}> is a pure strategy Nash equilibrium.'.format(psne[0][0], psne[0][1]))
    elif len(psne) > 1 and show_steps:
        print('{} are pure strategy Nash equilibria.'.format(', '.join(['<{}, {}>'.format(s1, s2) for s1, s2 in psne])))
    return psne


def IESDS(payoff_matrix, show_steps=True):
    # get player 1's and player 2's dominated strategies
    p1_dominated_strategies = payoff_matrix.dominated_strategies(player=1)
    p2_dominated_strategies = payoff_matrix.dominated_strategies(player=2)
    # print initial payoff matrix
    if show_steps:
        payoff_matrix.output()
        print()
    # iterate while there are dominated strategies to be eliminated
    while len(p1_dominated_strategies) > 0 or len(p2_dominated_strategies) > 0:
        if len(p1_dominated_strategies) > 0:
            # get first dominated player 1 strategy in list and eliminate it
            dominated_strategy = list(p1_dominated_strategies)[0]
            payoff_matrix.eliminate_strategy(player=1, strategy=dominated_strategy)
            # print more information
            if show_steps:
                dominating_strategy = p1_dominated_strategies[dominated_strategy]
                print("Player 1's strategy {} is strictly dominated by {}.\n\n".format(dominated_strategy, dominating_strategy))
        else:
            # get first dominated player 2 strategy in list and eliminate it
            dominated_strategy = list(p2_dominated_strategies)[0]
            payoff_matrix.eliminate_strategy(player=2, strategy=dominated_strategy)
            # print more information
            if show_steps:
                dominating_strategy = p2_dominated_strategies[dominated_strategy]
                print("Player 2's strategy {} is strictly dominated by {}.\n\n".format(dominated_strategy, dominating_strategy))
        # update player 1's and player 2's dominated strategies
        p1_dominated_strategies = payoff_matrix.dominated_strategies(player=1)
        p2_dominated_strategies = payoff_matrix.dominated_strategies(player=2)
        # print payoff matrix
        if show_steps:
            payoff_matrix.output()
            print()
    # if there is only player 1 strategy and player 2 strategy left, return them
    if len(payoff_matrix.p1_strategies) == 1 and len(payoff_matrix.p2_strategies) == 1:
        p1_strategy = payoff_matrix.p1_strategies[0]
        p2_strategy = payoff_matrix.p2_strategies[0]
        if show_steps:
            print('<{}, {}> is a pure strategy Nash equilibrium.'.format(p1_strategy, p2_strategy))
        return [(p1_strategy, p2_strategy)]
    # otherwise: use best responses for the remaining payoff matrix
    if show_steps:
        print('There are no strictly dominated strategies left to eliminate. Continuing with best responses...\n\n')
    return best_responses(payoff_matrix, show_steps)
