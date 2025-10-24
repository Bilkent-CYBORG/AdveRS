import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def find_x_ro_action(X, Y, dist_metric=euclidean_distances, r_ro=1):
    best_index = []
    best_min_value = (
        -np.inf
    )  # Initialize with negative infinity to find the largest minimum value

    distance_matrix = dist_metric(X, X)
    indices_in_epsilon = distance_matrix <= r_ro

    for index in range(X.shape[0]):
        # Find the minimum value of Y within the epsilon ball
        min_value_in_ball = np.min(Y[indices_in_epsilon[index]])

        # Update best index and minimum value if a new larger minimum value is found
        if min_value_in_ball > best_min_value:
            best_min_value = min_value_in_ball
            best_index = [index]
        elif min_value_in_ball == best_min_value:
            best_index.append(index)

    return np.random.choice(best_index)


def find_x_ro_action_diabetes_contextual(
    C, Y, ind_ref_cho, dist_metric=euclidean_distances, r_ro=1
):
    """
    Perform adversarial Robust Optimization (RO) in the contextual setting.

    Parameters:
    Y : n by m array-like (Y_mat[:, :, ind_ref_fbg])
        The array of target variables for every cho value.
    ind_ref_cho : int
        The index of the reference cho value.
    r_ro : float
        Radius of the epsilon ball.
    dist_metric : function, optional
        The distance metric to use (default is scipy.spatial.distance.euclidean).

    Returns:
    best_index : Index of the RO action (integer)
    """

    distance_matrix = dist_metric(C[ind_ref_cho].reshape(1, -1), C).flatten()
    indices_in_epsilon = distance_matrix <= r_ro

    min_value_in_ball_for_action = np.min(Y[:, indices_in_epsilon], axis=1)
    best_min_value = np.max(min_value_in_ball_for_action)
    min_value_indices = np.nonzero(min_value_in_ball_for_action == best_min_value)[0]
    return np.random.choice(min_value_indices)


def get_epsilon_ball(X, x_center, r_ro=0.1, dist_metric=euclidean_distances):
    distances = dist_metric(X, x_center.reshape(1, -1), squared=True)
    indices = distances <= r_ro
    return indices
