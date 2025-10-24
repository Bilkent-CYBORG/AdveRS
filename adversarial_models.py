import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def worst_case_attack(index_action, X, Y, epsilon, dist_metric=euclidean_distances):
    """
    Perform an worst case attack on the action point.

    Parameters:
    index_action (int): The index of the action point to perform the attack on.
    X (array-like): The input data points.
    Y (array-like): The corresponding output labels.
    epsilon (float): The perturbation budget of the adversary.
    dist_metric (function, optional): The distance metric to use for calculating distances.
        Defaults to `distance.euclidean`.
    Returns:
    index_x_perturbed (int): The index of the point with the minimum Y value within the epsilon ball.
    delta_x_perturbed (float): The distance from the action point to the perturbed point.
    """

    # Calculate distances from the action point
    # distances = np.array([dist_metric(X[index_action], x) for x in X])
    distances = dist_metric(X, X[index_action].reshape(1, -1))

    # Find points within the epsilon ball using atleast_1d to avoid deprecation warning
    within_epsilon = np.atleast_1d(distances) <= epsilon

    # Get the index of values within the epsilon ball
    Y_in_epsilon_ball = Y[within_epsilon]
    index_in_epsilon_ball = np.where(within_epsilon)[0]

    # Find the index of the point with the minimum Y value within the epsilon ball
    index_x_perturbed = index_in_epsilon_ball[np.argmin(Y_in_epsilon_ball)]
    delta_x_perturbed = distances[index_x_perturbed]

    return index_x_perturbed, delta_x_perturbed


def random_attack(index_action, X, Y, epsilon, dist_metric=euclidean_distances):
    """
    Perform a random attack on the action point.

    Parameters:
    index_action (int): The index of the action point to perform the attack on.
    X (array-like): The input data points.
    Y (array-like): The corresponding output labels.
    epsilon (float): The perturbation budget of the adversary.
    dist_metric (function, optional): The distance metric to use for calculating distances.
        Defaults to `distance.euclidean`.
    Returns:
    index_x_perturbed (int): The index of the point with the minimum Y value within the epsilon ball.
    delta_x_perturbed (float): The distance from the action point to the perturbed point.
    """
    # To avoid unkown error
    if epsilon <= 0:
        epsilon = 1e-6

    # Calculate distances from the action point
    # distances = np.array([dist_metric(X[index_action], x) for x in X])
    distances = dist_metric(X, X[index_action].reshape(1, -1))

    # Find points within the epsilon ball using atleast_1d to avoid deprecation warning
    within_epsilon = np.atleast_1d(distances) <= epsilon

    # Get the index of values within the epsilon ball
    index_in_epsilon_ball = np.where(within_epsilon)[0]

    # Choose a random point within the epsilon ball
    index_x_perturbed = np.random.choice(index_in_epsilon_ball)
    delta_x_perturbed = distances[index_x_perturbed]

    return index_x_perturbed, delta_x_perturbed
