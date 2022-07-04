import numpy as np
from sympy import symbols, Eq, solve


def __MSNE(player, payoff_matrix, show_steps=True):
    # extract player's payoffs from payoff matrix
    players_payoffs = payoff_matrix.payoffs[:, :, player-1]
    # transpose player 2's payoffs to use the same order in calculation
    if player == 2:
        players_payoffs = players_payoffs.transpose()
    if show_steps:
        # define variables for easier use in formatting
        opponent = 3 - player
        if player == 2:
            players_strategies = payoff_matrix.p2_strategies
            opponents_strategies = payoff_matrix.p1_strategies
        else:
            players_strategies = payoff_matrix.p1_strategies
            opponents_strategies = payoff_matrix.p2_strategies
        # print steps for solving the equation
        print('Player {0} plays {1} and {2} randomly with probabilities σ_{1} and σ_{2}:\n'.format(opponent, *opponents_strategies))
        for i, strategy in enumerate(players_strategies):
            print("Player {}'s expected utility for playing {}:".format(player, strategy))
            print('EU_{0} = σ_{1} * {3} + σ_{2} * {4}\n'.format(strategy, *opponents_strategies, *players_payoffs[i]))
        print('Solve for σ_{} and σ_{}:'.format(*opponents_strategies))
        print('(I)  EU_{} = EU_{}'.format(*players_strategies))
        print('     σ_{0} * {2} + σ_{1} * {3} = σ_{0} * {4} + σ_{1} * {5}'.format(*opponents_strategies, *players_payoffs.flatten()))
        print('(II) σ_{} + σ_{} = 1'.format(*opponents_strategies))
    # solve for prob_1 and prob_2
    prob_1, prob_2 = symbols('prob_1 prob_2')
    EU_A = prob_1 * players_payoffs[0, 0] + prob_2 * players_payoffs[0, 1]
    EU_B = prob_1 * players_payoffs[1, 0] + prob_2 * players_payoffs[1, 1]
    eq1 = Eq(EU_A, EU_B)
    eq2 = Eq(prob_1 + prob_2, 1)
    solutions = solve((eq1, eq2), (prob_1, prob_2))
    if show_steps:
        print('->   σ_{0} = {2}\n     σ_{1} = {3}\n'.format(*opponents_strategies, *solutions.values()))
    # return MSNE if no probabilities are negative
    MSNE = float(solutions[prob_1]), float(solutions[prob_2])
    if MSNE[0] >= 0 and MSNE[1] >= 0:
        print('The mixed strategy Nash equilibrium is <{}, {}>.'.format(solutions[prob_1], solutions[prob_2]))
        return MSNE
    elif show_steps:
        print('There is no mixed strategy Nash equilibrium.')


def mixed_strategies(payoff_matrix, show_steps=True):
    # skip if payoff matrix is not of size 2x2
    if payoff_matrix.payoffs.shape[0] != 2 or payoff_matrix.payoffs.shape[1] != 2:
        return
    # initialize list of MSNE for output
    MSNEs = []
    # iterate over players, determine the MSNE and add it to list
    for player in range(1, 3):
        MSNE = __MSNE(player, payoff_matrix, show_steps)
        if MSNE is not None:
            MSNEs.append(MSNE)
        if show_steps and player != 2:
            print('\n')
    # return list of mixed strategy Nash equilibria
    return MSNEs


def __EU(player, payoff_matrix, MSNE_1, MSNE_2, show_steps=True):
    # extract player's payoffs from payoff matrix
    players_payoffs = payoff_matrix.payoffs[:, :, player-1]
    # compute expected utility by multiplying probabilities with payoffs
    MSNEs = np.array([MSNE_1, MSNE_2])
    EU = np.sum(np.outer(MSNEs[0], MSNEs[1]) * players_payoffs)
    if show_steps:
        # if possible, convert all values to integers for prettier output
        if np.all(players_payoffs % 1 == 0):
            players_payoffs = players_payoffs.astype(np.int)
        # define variables to make lines shorter
        payoffs_flat = players_payoffs.flatten()
        MSNEs_flat = MSNEs.flatten()
        # print calculation steps
        print("Player {}'s overall expected utility:".format(player))
        print('EU_{0} = {5} * {7} * {1} + {5} * {8} * {2} + {6} * {7} * {3} + {6} * {8} * {4}'.format(player, *payoffs_flat, *MSNEs_flat))
        print('EU_{} = {}'.format(player, EU))
    return EU


def expected_utilities(payoff_matrix, MSNE_1, MSNE_2, show_steps=True):
    # skip if payoff matrix is not of size 2x2
    if payoff_matrix.payoffs.shape[0] != 2 or payoff_matrix.payoffs.shape[1] != 2:
        return
    # initialize list of MSNE for output
    EUs = []
    # iterate over players, determine the expected utility and add it to list
    for player in range(1, 3):
        EU = __EU(player, payoff_matrix, MSNE_1, MSNE_2, show_steps)
        EUs.append(EU)
        if show_steps and player != 2:
            print('\n')
    # return list of expected utilities
    return EUs
