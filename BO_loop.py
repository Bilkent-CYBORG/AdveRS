import GPy
import numpy as np
from functools import partial
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances

from adversarial_models import *
from adv_RO import find_x_ro_action, find_x_ro_action_diabetes_contextual


def BO_loop_RS(
    X,
    Y,
    acquisition_function,
    attack_mode=0,
    distance_metric=distance.euclidean,
    T=50,
    beta_t=2,
    sigma_noise=1e-10,
    num_iter=1,
    num_init_samples=10,
    kernel=GPy.kern.RBF(1),
    tau=1,
    r_ro=None,
    epsilon_t=0.1,
    print_samples=False,
):
    """
    Bayesian optimization loop.

    Parameters
    ----------
    X : np.ndarray
        Initial input samples.
    Y : np.ndarray
        Initial output samples.
    T : int
        Time horizon.
    num_iter : int
        Number of iterations to run the optimization loop.
    num_init_samples : int
        Number of random initial samples.
    kernel : GPy.kern
        Kernel function.

    Returns
    -------
    to_return: np.ndarray of size (num_iter, 2, T) of index_x_t, index_tilde_x_t

    dataset : Tuple of samples.
        dataset[0] : np.ndarray of acquisition points in X.
        dataset[1] : np.ndarray of noise samples of Y at perturbed points
        dataset[2] : np.ndarray of indicies of acquisition points.
        dataset[3] : np.ndarray of indicies of perturbed points.
    """
    # Initialize an empty list to hold datasets for each iteration
    to_return = []

    # Initialize acquisition function
    if r_ro is not None:
        acquisition = partial(
            acquisition_function, X=X, tau=tau, r_ro=r_ro, dist_metric=distance_metric
        )
    else:
        acquisition = partial(
            acquisition_function, X=X, tau=tau, dist_metric=distance_metric
        )

    # Initialize pertubation budget array
    if isinstance(epsilon_t, (int, float)):
        epsilon_t = epsilon_t * np.ones(T)
    elif isinstance(epsilon_t, np.ndarray):
        print("Array for perturbation budget is provided")
    else:
        Warning("epsilon_t is neither a number nor an array")

    print(
        f"Starting Bayesian optimization loop with {num_iter} iterations and {T} rounds \n"
    )

    # Bayesian optimization loop
    for iter in range(num_iter):
        print("Iteration: ", iter)

        # Initialize Data
        X_init_index = np.random.choice(X.shape[0], num_init_samples, replace=False)
        Y_init = Y[X_init_index]
        current_dataset = (
            np.array(X[X_init_index]),
            np.array(Y_init),
            np.array([]),
            np.array([]),
        )

        # Initialize GP model
        model = GPy.models.GPRegression(
            current_dataset[0], current_dataset[1], kernel, noise_var=sigma_noise**2
        )
        mu_t, var_t = model.predict(X, include_likelihood=False)
        sigma_t = np.sqrt(var_t)

        for t in range(T):
            # Compute the ucb and lcb for all points
            ucb_t = mu_t + beta_t * sigma_t
            lcb_t = mu_t - beta_t * sigma_t

            # Find the index of acquisiton point
            index_tilde_x_t, bar_kappa_tau_t = acquisition(Y=ucb_t)

            # Adversary picks perturbed point
            if attack_mode == 0:
                index_x_t, delta_t = random_attack(
                    index_action=index_tilde_x_t,
                    X=X,
                    Y=lcb_t,
                    epsilon=epsilon_t[t],
                    dist_metric=distance_metric,
                )
            elif attack_mode == 1:
                index_x_t, delta_t = worst_case_attack(
                    index_action=index_tilde_x_t,
                    X=X,
                    Y=lcb_t,
                    epsilon=epsilon_t[t],
                    dist_metric=distance_metric,
                )
            elif attack_mode == 2:
                index_x_t, delta_t = worst_case_attack(
                    index_action=index_tilde_x_t,
                    X=X,
                    Y=Y,
                    epsilon=epsilon_t[t],
                    dist_metric=distance_metric,
                )
            elif attack_mode == 3:
                gaussian_perturbation = (
                    np.random.multivariate_normal(
                        np.zeros(X.shape[1]), np.eye(X.shape[1])
                    )
                    * epsilon_t[t]
                )
                tilde_x_t = X[index_tilde_x_t]
                index_x_t = np.argmin(
                    np.linalg.norm(X - (tilde_x_t + gaussian_perturbation), axis=1)
                )
            else:
                raise ValueError("Invalid attack mode")

            # Evaluate the objective function
            y_t = Y[index_x_t] + np.random.normal(0, sigma_noise)

            # Update the dataset
            current_dataset = (
                np.append(current_dataset[0], X[index_x_t]),
                np.append(current_dataset[1], y_t),
                np.append(current_dataset[2], index_tilde_x_t),
                np.append(current_dataset[3], index_x_t),
            )

            # Update the GP model
            model.set_XY(
                X=current_dataset[0].reshape(-1, X.shape[1]),
                Y=current_dataset[1].reshape(-1, 1),
            )

            # Update the GP model predictions
            mu_t, sigma_t = model.predict(X, include_likelihood=False)
            if print_samples and t % 5 == 0:
                print("Round: ", t)
                print(f"X_acquisition: {X[int(current_dataset[2][t])][0]:.3f}")
                print("-----------------")

        # Append the current dataset to the list of all datasets
        to_return.append((current_dataset[2], current_dataset[3]))

    # Convert lists to numpy arrays for easier averaging later
    to_return = np.array(to_return)
    print("Bayesian optimization loop completed \n")
    return to_return


def BO_loop_RO(
    X,
    Y,
    attack_mode=0,
    distance_metric=euclidean_distances,
    T=50,
    beta_t=2,
    sigma_noise=1e-10,
    num_iter=1,
    num_init_samples=10,
    kernel=GPy.kern.RBF(1),
    r_ro=0.1,
    epsilon_t=0.1,
    print_samples=False,
):
    """
    Bayesian optimization loop.

    Parameters
    ----------
    X : np.ndarray
        Initial input samples.
    Y : np.ndarray
        Initial output samples.
    T : int
        Time horizon.
    num_iter : int
        Number of iterations to run the optimization loop.
    num_init_samples : int
        Number of random initial samples.
    kernel : GPy.kern
        Kernel function.

    Returns
    -------
    dataset : Tuple of samples.
        dataset[0] : np.ndarray of acquisition points in X.
        dataset[1] : np.ndarray of noise samples of Y at perturbed points
        dataset[2] : np.ndarray of indicies of acquisition points.
        dataset[3] : np.ndarray of indicies of perturbed points.
    """
    # Initialize an empty list to hold datasets for each iteration
    to_return = []

    # Initialize pertubation budget array
    if isinstance(epsilon_t, (int, float)):
        epsilon_t = epsilon_t * np.ones(T)
    elif isinstance(epsilon_t, np.ndarray):
        pass
    else:
        Warning("epsilon_t is neither a number nor an array")

    # Initialize the radius of the RO array
    if isinstance(r_ro, (int, float)):
        r_ro = r_ro * np.ones(T)
    elif isinstance(r_ro, np.ndarray):
        pass
    else:
        Warning("r_ro is neither a number nor an array")

    print(
        f"Starting Bayesian optimization loop with {num_iter} iterations and {T} rounds \n"
    )

    # Bayesian optimization loop
    for iter in range(num_iter):
        print("Iteration: ", iter)

        # Initialize Data
        X_init_index = np.random.choice(X.shape[0], num_init_samples, replace=False)
        Y_init = Y[X_init_index]
        current_dataset = (
            np.array(X[X_init_index]),
            np.array(Y_init),
            np.array([]),
            np.array([]),
        )

        # Initialize GP model
        model = GPy.models.GPRegression(
            current_dataset[0], current_dataset[1], kernel, noise_var=sigma_noise**2
        )
        mu_t, var_t = model.predict(X, include_likelihood=False)
        sigma_t = np.sqrt(var_t)

        for t in range(T):
            # Set the acquisition function with radius of RO
            acquisition = partial(
                find_x_ro_action, X=X, Y=Y, r_ro=r_ro[t], dist_metric=distance_metric
            )

            # Compute the ucb and lcb for all points
            ucb_t = mu_t + beta_t * sigma_t
            lcb_t = mu_t - beta_t * sigma_t

            # Find the index of acquisiton point
            index_tilde_x_t = acquisition(Y=ucb_t)

            # Adversary picks perturbed point
            if attack_mode == 0:
                index_x_t, delta_t = random_attack(
                    index_action=index_tilde_x_t,
                    X=X,
                    Y=lcb_t,
                    epsilon=epsilon_t[t],
                    dist_metric=distance_metric,
                )
            elif attack_mode == 1:
                index_x_t, delta_t = worst_case_attack(
                    index_action=index_tilde_x_t,
                    X=X,
                    Y=lcb_t,
                    epsilon=epsilon_t[t],
                    dist_metric=distance_metric,
                )
            elif attack_mode == 2:
                index_x_t, delta_t = worst_case_attack(
                    index_action=index_tilde_x_t,
                    X=X,
                    Y=Y,
                    epsilon=epsilon_t[t],
                    dist_metric=distance_metric,
                )
            elif attack_mode == 3:
                gaussian_perturbation = (
                    np.random.multivariate_normal(
                        np.zeros(X.shape[1]), np.eye(X.shape[1])
                    )
                    * epsilon_t[t]
                )
                tilde_x_t = X[index_tilde_x_t]
                index_x_t = np.argmin(
                    np.linalg.norm(X - (tilde_x_t + gaussian_perturbation), axis=1)
                )
            else:
                raise ValueError("Invalid attack mode")

            # Evaluate the objective function
            y_t = Y[index_x_t] + np.random.normal(0, sigma_noise)

            # Update the dataset
            current_dataset = (
                np.append(current_dataset[0], X[index_x_t]),
                np.append(current_dataset[1], y_t),
                np.append(current_dataset[2], index_tilde_x_t),
                np.append(current_dataset[3], index_x_t),
            )

            # Update the GP model
            model.set_XY(
                X=current_dataset[0].reshape(-1, X.shape[1]),
                Y=current_dataset[1].reshape(-1, 1),
            )

            # Update the GP model predictions
            mu_t, sigma_t = model.predict(X, include_likelihood=False)
            if print_samples and t % 5 == 0:
                print("Round: ", t)
                print(f"X_acquisition: {X[int(current_dataset[2][t])][0]:.3f}")
                print("-----------------")

        # Append the current dataset to the list of all datasets (tilde_x_t, x_t)
        to_return.append((current_dataset[2], current_dataset[3]))

    # Convert lists to numpy arrays for easier averaging later
    to_return = np.array(to_return)

    print("Bayesian optimization loop completed \n")
    return to_return


def BO_loop_RO_diabetes(
    X,
    Y,
    C_ref_inds,
    C_true_inds,
    fbg_true_inds,
    distance_metric=euclidean_distances,
    T=50,
    beta_t=2,
    sigma_noise=1e-10,
    num_iter=1,
    num_init_samples=10,
    kernel=GPy.kern.RBF(1),
    r_ro=0.1,
    print_samples=False,
):
    """
    Bayesian optimization loop.

    Parameters
    ----------
    X : n*m*k by 3 np.ndarray
        Initial input samples.
    Y : n by m by k np.ndarray
        Initial output samples.
    C_ref_inds : num_iter by T np.ndarray
        Reference cho values.
    C_true_inds : num_iter by T np.ndarray
        True cho values.
    fbg_true_inds : num_iter by T np.ndarray
        True fbg values.
    T : int
        Time horizon.
    num_iter : int
        Number of iterations to run the optimization loop.
    num_init_samples : int
        Number of random initial samples.
    kernel : GPy.kern
        Kernel function.

    Returns
    -------
    dataset : Tuple of samples.
        dataset[0] : np.ndarray of acquisition points in X.
        dataset[1] : np.ndarray of noise samples of Y at perturbed points
        dataset[2] : np.ndarray of indicies of acquisition points.
        dataset[3] : np.ndarray of indicies of perturbed points.
    """
    n, m, k = Y.shape
    Y_flat = Y.reshape(n * m * k, 1)
    X_mat = X.reshape(n, m, k, 3)
    C = X_mat[0, :, 0, 1].reshape(-1, 1)  # distinct cho values

    # Initialize an empty list to hold datasets for each iteration
    to_return = []

    # Initialize the radius of the RO array
    if isinstance(r_ro, (int, float)):
        r_ro = r_ro * np.ones(T)
    elif isinstance(r_ro, np.ndarray):
        pass
    else:
        Warning("r_ro is neither a number nor an array")

    print(
        f"Starting Bayesian optimization loop with {num_iter} iterations and {T} rounds \n"
    )

    # Bayesian optimization loop
    for iter in range(num_iter):
        print("Iteration: ", iter)

        # Initialize Data
        X_init_index = np.random.choice(X.shape[0], num_init_samples, replace=False)
        Y_init = Y_flat[X_init_index]
        current_dataset = (
            np.array(X[X_init_index]),
            np.array(Y_init),
            np.array([]),
            np.array([]),
        )

        # Initialize GP model
        model = GPy.models.GPRegression(
            current_dataset[0], current_dataset[1], kernel, noise_var=sigma_noise**2
        )
        mu_t, var_t = model.predict(X, include_likelihood=False)
        sigma_t = np.sqrt(var_t)

        for t in range(T):
            # Compute the ucb and lcb for all points
            ucb_t = mu_t + beta_t * sigma_t
            lcb_t = mu_t - beta_t * sigma_t

            ucb_t_mat = ucb_t.reshape(n, m, k)

            # Set the acquisition function with radius of RO
            acquisition = partial(
                find_x_ro_action_diabetes_contextual,
                C=C,
                ind_ref_cho=C_ref_inds[iter, t],
                r_ro=r_ro[t],
                dist_metric=distance_metric,
            )

            # Find the index of acquisiton point
            index_tilde_ins_t = acquisition(Y=ucb_t_mat[:, :, fbg_true_inds[iter, t]])

            # True point
            index_x_t = (
                (index_tilde_ins_t * (m * k))
                + (C_true_inds[iter, t] * k)
                + fbg_true_inds[iter, t]
            )
            index_x_tilde_t = (
                (index_tilde_ins_t * (m * k))
                + (C_ref_inds[iter, t] * k)
                + fbg_true_inds[iter, t]
            )

            # Evaluate the objective function
            y_t = Y_flat[index_x_t] + np.random.normal(0, sigma_noise)

            # Update the dataset
            current_dataset = (
                np.append(current_dataset[0], X[index_x_t]),
                np.append(current_dataset[1], y_t),
                np.append(current_dataset[2], index_x_tilde_t),
                np.append(current_dataset[3], index_x_t),
            )

            # Update the GP model
            model.set_XY(
                X=current_dataset[0].reshape(-1, X.shape[1]),
                Y=current_dataset[1].reshape(-1, 1),
            )

            # Update the GP model predictions
            mu_t, sigma_t = model.predict(X, include_likelihood=False)
            if print_samples and t % 5 == 0:
                print("Round: ", t)
                print(f"X_acquisition: {X[int(current_dataset[2][t])][0]:.3f}")
                print("-----------------")

        # Append the current dataset to the list of all datasets (tilde_x_t, x_t)
        to_return.append((current_dataset[2], current_dataset[3]))

    # Convert lists to numpy arrays for easier averaging later
    to_return = np.array(to_return)

    print("Bayesian optimization loop completed \n")
    return to_return


def BO_loop_RS_diabetes(
    X,
    Y,
    C_ref_inds,
    C_true_inds,
    fbg_true_inds,
    acquisition_function,
    distance_metric=euclidean_distances,
    T=50,
    beta_t=2,
    sigma_noise=1e-10,
    num_iter=1,
    num_init_samples=10,
    kernel=GPy.kern.RBF(1),
    tau=0.1,
    print_samples=False,
):
    """
    Bayesian optimization loop.

    Parameters
    ----------
    X : n*m*k by 3 np.ndarray
        Initial input samples.
    Y : n by m by k np.ndarray
        Initial output samples.
    C_ref_inds : num_iter by T np.ndarray
        Reference cho values.
    C_true_inds : num_iter by T np.ndarray
        True cho values.
    fbg_true_inds : num_iter by T np.ndarray
        True fbg values.
    T : int
        Time horizon.
    num_iter : int
        Number of iterations to run the optimization loop.
    num_init_samples : int
        Number of random initial samples.
    kernel : GPy.kern
        Kernel function.

    Returns
    -------
    dataset : Tuple of samples.
        dataset[0] : np.ndarray of acquisition points in X.
        dataset[1] : np.ndarray of noise samples of Y at perturbed points
        dataset[2] : np.ndarray of indicies of acquisition points.
        dataset[3] : np.ndarray of indicies of perturbed points.
    """
    n, m, k = Y.shape
    Y_flat = Y.reshape(n * m * k, 1)
    X_mat = X.reshape(n, m, k, 3)
    C = X_mat[0, :, 0, 1].reshape(-1, 1)  # distinct cho values

    # Initialize an empty list to hold datasets for each iteration
    to_return = []

    # Initialize acquisition function
    acquisition = partial(
        acquisition_function, C=C, tau=tau, dist_metric=distance_metric
    )

    print(
        f"Starting Bayesian optimization loop with {num_iter} iterations and {T} rounds \n"
    )

    # Bayesian optimization loop
    for iter in range(num_iter):
        print("Iteration: ", iter)

        # Initialize Data
        X_init_index = np.random.choice(X.shape[0], num_init_samples, replace=False)
        Y_init = Y_flat[X_init_index]
        current_dataset = (
            np.array(X[X_init_index]),
            np.array(Y_init),
            np.array([]),
            np.array([]),
        )

        # Initialize GP model
        model = GPy.models.GPRegression(
            current_dataset[0], current_dataset[1], kernel, noise_var=sigma_noise**2
        )
        mu_t, var_t = model.predict(X, include_likelihood=False)
        sigma_t = np.sqrt(var_t)

        for t in range(T):
            # Compute the ucb and lcb for all points
            ucb_t = mu_t + beta_t * sigma_t
            lcb_t = mu_t - beta_t * sigma_t

            ucb_t_mat = ucb_t.reshape(n, m, k)

            # Find the index of acquisiton point
            index_tilde_ins_t, bar_kappa_tau_t = acquisition(
                Y=ucb_t_mat[:, :, fbg_true_inds[iter, t]],
                ind_ref_cho=C_ref_inds[iter, t],
            )

            # True point
            index_x_t = (
                (index_tilde_ins_t * (m * k))
                + (C_true_inds[iter, t] * k)
                + fbg_true_inds[iter, t]
            )
            index_x_tilde_t = (
                (index_tilde_ins_t * (m * k))
                + (C_ref_inds[iter, t] * k)
                + fbg_true_inds[iter, t]
            )

            # Evaluate the objective function
            y_t = Y_flat[index_x_t] + np.random.normal(0, sigma_noise)

            # Update the dataset
            current_dataset = (
                np.append(current_dataset[0], X[index_x_t]),
                np.append(current_dataset[1], y_t),
                np.append(current_dataset[2], index_x_tilde_t),
                np.append(current_dataset[3], index_x_t),
            )

            # Update the GP model
            model.set_XY(
                X=current_dataset[0].reshape(-1, X.shape[1]),
                Y=current_dataset[1].reshape(-1, 1),
            )

            # Update the GP model predictions
            mu_t, sigma_t = model.predict(X, include_likelihood=False)
            if print_samples and t % 5 == 0:
                print("Round: ", t)
                print(f"X_acquisition: {X[int(current_dataset[2][t])][0]:.3f}")
                print("-----------------")

        # Append the current dataset to the list of all datasets (tilde_x_t, x_t)
        to_return.append((current_dataset[2], current_dataset[3]))

    # Convert lists to numpy arrays for easier averaging later
    to_return = np.array(to_return)

    print("Bayesian optimization loop completed \n")
    return to_return
