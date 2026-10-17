import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def find_x_rs_action_RS1(X, Y, tau, dist_metric=euclidean_distances, pow=1):
    """
    Perform adversarial Robust Satisficing (RS-G) with p=pow. (when p=1, RS-1)

    Parameters:
    X : n by 1 array-like
        The array of input variables.
    Y : n by 1 array-like
        The array of target variables.
    tau : float
        The threshold value for the satisficing condition.
    dist_metric : function, optional
        The distance metric to use (default is scipy.spatial.distance.euclidean).
    pow : float, optional
        Smoothness parameter (default is 1).

    Returns:
    result : tuple of index_x_rs and kappa_x_rs
    """

    X_less_than_tau_index = np.where(Y < tau)[0]
    X_more_than_tau_index = np.where(Y >= tau)[0]
    N = len(X)
    kappa_values = np.inf * np.ones(N)

    # If all points are above or below tau, return a random point
    if len(X_less_than_tau_index) == 0:
        print("Function above tau")
        return np.random.choice(N), -np.inf
    elif len(X_more_than_tau_index) == 0:
        print("Function below tau")
        return np.random.choice(N), np.inf

    # Initialize kappa_values based on the condition
    kappa_values[X_more_than_tau_index] = -np.inf
    kappa_values[X_less_than_tau_index] = np.inf

    distances = dist_metric(X[X_more_than_tau_index], X[X_less_than_tau_index])

    for i, index_more in enumerate(X_more_than_tau_index):

        kappa_temp_values = (
            np.power((tau - Y[X_less_than_tau_index].flatten()), 1 / pow) / distances[i]
        )
        kappa_values[index_more] = np.max(kappa_temp_values)

    index_x_rs = np.argmin(kappa_values)
    kappa_x_rs = kappa_values[index_x_rs]
    return index_x_rs, kappa_x_rs


def find_x_rs_action_RS1_diabetes_contextual(
    C, Y, ind_ref_cho, tau, dist_metric=euclidean_distances, pow=1
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
    pow : float, optional
        Smoothness parameter (default is 1).

    Returns:
    result : tuple of index_x_rs and kappa_x_rs
    """

    # Find the indices of the target variables that are less than tau at the reference cho value
    X_less_than_tau_index = np.where(Y[:, ind_ref_cho] < tau)[0]
    X_more_than_tau_index = np.where(Y[:, ind_ref_cho] >= tau)[0]
    N_x = Y.shape[0]
    kappa_values = np.inf * np.ones(N_x)

    # If all points are below tau, return a random point
    if len(X_less_than_tau_index) == N_x:
        print("Function below tau")
        return np.random.choice(N_x), np.inf

    # Initialize kappa_values based on the condition
    kappa_values[X_more_than_tau_index] = -np.inf
    kappa_values[X_less_than_tau_index] = np.inf

    # Distances between the context of the reference cho value and the context of the other cho values
    distances = (
        dist_metric(C[ind_ref_cho].reshape(1, -1), C) + 1e-6
    )  # Adding a small value to avoid division by zero
    distances = np.power(distances, pow)

    # Method 1: Loop over X
    for i, index_more in enumerate(X_more_than_tau_index):
        kappa_x_temp = (tau - Y[index_more, :]) / distances.flatten()
        kappa_values[index_more] = np.max(kappa_x_temp)

    index_x_rs = np.argmin(kappa_values)
    kappa_x_rs = kappa_values[index_x_rs]

    return index_x_rs, np.power(kappa_x_rs, 1 / pow)


# This is faster than GPy built-in kernel function. Not being used now, if using check parameters
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
