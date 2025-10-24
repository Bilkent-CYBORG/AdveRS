import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances


def find_x_rs_action_RS2(X, Y, tau, dist_metric=euclidean_distances):
    """
    Perform adversarial Robust Satisficing (RS) with epsilon ball (RS-2).

    Parameters:
    Y : n by 1 array-like
        The array of target variables.
    X : n by 1 array-like
        The array of input variables.
    tau : float
        The threshold value for the satisficing condition.
    dist_metric : function, optional
        The distance metric to use (default is scipy.spatial.distance.euclidean).

    Returns:
    result : tuple of index_x_rs and eps_x_rs

    """

    X_more_than_tau_index = np.where(Y >= tau)[0]
    X_less_than_tau_index = np.where(Y < tau)[0]
    N = len(X)
    epsilon_values = np.inf * np.ones(N)

    # If all points are above or below tau, return a random point
    if len(X_less_than_tau_index) == 0:
        print("Function above tau")
        return np.random.choice(N), np.inf
    elif len(X_more_than_tau_index) == 0:
        print("Function below tau")
        return np.random.choice(N), -np.inf

    # Initialize kappa_values based on the condition
    epsilon_values[X_more_than_tau_index] = +np.inf
    epsilon_values[X_less_than_tau_index] = -np.inf

    distances = dist_metric(X[X_more_than_tau_index], X[X_less_than_tau_index])

    for i, index_more in enumerate(X_more_than_tau_index):
        epsilon_values[index_more] = np.min(distances[i])

    index_x_rs = np.argmax(epsilon_values)
    epsilon_x_rs = epsilon_values[index_x_rs]
    return index_x_rs, epsilon_x_rs


def find_x_rs_action_RS2_diabetes_contextual(
    C, Y, ind_ref_cho, tau, dist_metric=euclidean_distances
):
    """
    Perform adversarial Robust Satisficing (RS) in the contextual setting.

    Parameters:
    Y : n by m array-like (Y_mat[:, :, ind_ref_fbg])
        The array of target variables for every cho value.
    ind_ref_cho : int
        The index of the reference cho value.
    tau : float
        The threshold value for the satisficing condition.
    dist_metric : function, optional
        The distance metric to use (default is scipy.spatial.distance.euclidean).

    Returns:
    result : tuple of index_x_rs and epsilon_x_rs
    """

    # Find the indices of the target variables that are less than tau at the reference cho value
    X_less_than_tau_index = np.where(Y[:, ind_ref_cho] < tau)[0]
    X_more_than_tau_index = np.where(Y[:, ind_ref_cho] >= tau)[0]
    N_x = Y.shape[0]
    epsilon_values = np.inf * np.ones(N_x)

    # If all points are below tau, return a random point
    if len(X_less_than_tau_index) == N_x:
        print("Function below tau")
        return np.random.choice(N_x), -np.inf

    # Initialize kappa_values based on the condition
    epsilon_values[X_more_than_tau_index] = np.inf
    epsilon_values[X_less_than_tau_index] = -np.inf

    # Distances between the context of the reference cho value and the context of the other cho values
    distances = dist_metric(C[ind_ref_cho].reshape(1, -1), C).flatten()

    for i, index_more in enumerate(X_more_than_tau_index):
        C_less_than_tau_index = np.where(Y[index_more, :] < tau)[0]
        if len(C_less_than_tau_index) == 0:
            epsilon_values[index_more] = np.inf
        else:
            epsilon_values[index_more] = np.min(distances[C_less_than_tau_index])

    max_epsilon = np.max(epsilon_values)
    max_epsilon_indices = np.nonzero(epsilon_values == max_epsilon)[0]
    index_x_rs = np.random.choice(max_epsilon_indices)
    epsilon_x_rs = epsilon_values[index_x_rs]
    return index_x_rs, epsilon_x_rs


# This is faster than GPy buit-in kernel function. Not being used now, if using check parameters
def kernel_metric(x1, x2):
    """
    Kernel paramaters are inside RBF function
    """
    dist = np.sqrt(RBF(x1, x1) - 2 * RBF(x1, x2) + RBF(x2, x2))
    return dist


def RBF(x1, x2, variance=2.0, lengthscale=0.1):
    """
    RBF function
    """
    sqdist = np.sum((x1 - x2) ** 2)
    return variance * np.exp(-0.5 * sqdist / lengthscale**2)


def any_pt_in_eps_ball(X, X_indices, x_center, eps=0.1, dist_metric=distance.euclidean):
    """
    Checks if any point in X, indexed by X_indices, lies within an epsilon ball centered at x_center.

    Parameters:
    X (np.ndarray): Array of points.
    X_indices (list or np.ndarray): Indices of the points in X to be checked.
    x_center (np.ndarray): The center of the epsilon ball.
    eps (float): Radius of the epsilon ball. Default is 0.1.
    dist_metric (function): Distance metric to be used (default is Euclidean distance).

    Returns:
    bool: True if any point in X indexed by X_indices is within the epsilon ball centered at x_center, False otherwise.
    """
    for i in X_indices:
        if dist_metric(X[i].flatten(), x_center.flatten()) <= eps:
            return True
    return False
