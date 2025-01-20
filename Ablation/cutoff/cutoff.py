#inspired by Schaeffer et.al. 2023 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from numpy.polynomial import legendre
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.linalg

#%% parameters -------------------------------------------------------------------------------
n_test = 100
n_train_max  = 1000  # this has to be larger than 4**n_qubits
n_qubits = 3
reps = 5

#%% different functions to generate data ---------------------------------------------------------------

def generate_evenly_spaced_samples(low, high, n_test, n_qubits):
    # Calculate the number of grid points per dimension
    num_points_per_dim = int(np.ceil(n_test ** (1 / n_qubits)))

    # Generate evenly spaced points for each dimension
    linspaces = [np.linspace(start=low, stop=high, num=num_points_per_dim) for _ in range(n_qubits)]

    # Create a meshgrid for all dimensions
    grid = np.meshgrid(*linspaces, indexing='ij')

    # Stack the grid arrays along the last axis and reshape to a 2D array
    x_test_full = np.stack(grid, axis=-1).reshape(-1, n_qubits)

    # If there are more points than needed, select a subset of them
    if x_test_full.shape[0] > n_test:
        indices = np.linspace(0, x_test_full.shape[0] - 1, n_test, dtype=int)
        x_test = x_test_full[indices]
    else:
        x_test = x_test_full

    return x_test.astype(np.float64)

def generate_sine_data(n_samples, noise_level=0.0):
    x_data = np.random.rand(n_samples)*np.pi
    y_data = np.sin(1*x_data) + np.random.normal(0, noise_level, n_samples)
    return x_data.reshape(-1, 1), y_data

def generate_linear_data(n_samples, noise_level=0.05):
    x_data = np.random.rand(n_samples).astype(np.float64) * np.pi
    y_data = 2*x_data + np.random.normal(0, noise_level, n_samples).astype(np.float64)
    return x_data.reshape(-1, 1), y_data

def generate_multidim_linear_data(n_samples, noise_level=0.0):
    x_data = np.random.rand(n_samples, n_qubits).astype(np.float64)
    weights = np.arange(1, n_qubits + 1, dtype=np.float64)
    y_data = np.dot(x_data, weights) + np.random.normal(0, noise_level, n_samples).astype(np.float64)
    return x_data, y_data

def compute_y_from_x_multidim_linear(X):
    weights = np.arange(1, n_qubits + 1, dtype=np.float64)
    y = np.dot(X, weights)
    return y

def generate_linear_cos_data(n_samples, noise_level=0.001):
    x_data = np.random.uniform(-1, 1, size=(n_samples)).astype(np.float64)
    y_data = 2*x_data + np.cos(15*x_data) +  np.random.normal(0, noise_level, n_samples).astype(np.float64)
    return x_data.reshape(-1, 1), y_data

def compute_y_from_x(X: np.ndarray):
    return np.add(2.0 * X, np.cos(X * 25))[:, 0]

def generate_multidim_linear_cos_data(n_samples, noise_level=0.001):
    x_data = np.random.uniform(-1, 1, size=(n_samples, n_qubits)).astype(np.float64)
    weights_lin =  np.zeros(n_qubits)
    weights = np.arange(1, n_qubits + 1, dtype=np.float64) / n_qubits
    y_data = np.dot(x_data, weights_lin) + np.cos(3*np.dot(x_data, weights)) +  np.random.normal(0, noise_level, n_samples).astype(np.float64)
    return x_data, y_data

def generate_polynomial_data(n_samples, noise_level=0.001):
    x_data = np.random.uniform(-1, 1, size=(n_samples))
    y_data = -10*x_data - x_data**2 + 20*x_data**3 + np.random.normal(0, noise_level, n_samples)
    return x_data.reshape(-1, 1), y_data

noise_level = 0.0

low, high = -1.0, 1.0

scaler = StandardScaler()
scaler2 = MinMaxScaler(feature_range=(-np.pi/2, np.pi/2))

#%% some pre-computations -----------------------------------------------------------------------------
def quantum_feature_map(x):
    """
    Feature map for quantum kernel.
    Adjust repetitions (reps) or gates if kernel is not full rank.
    """
    reps = n_qubits
    for r in range(reps):
        for i in range(len(x)):
            qml.RX(x[i], wires=i)
        for n in range(n_qubits - 1):
            qml.CNOT(wires=[n, n+1])
        for i in range(len(x)):
            qml.RZ(x[i], wires=i)

dev_kernel = qml.device("default.qubit", wires=n_qubits)

projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
projector[0, 0] = 1

@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    quantum_feature_map(x1)
    qml.adjoint(quantum_feature_map)(x2)
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
        evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])

# compute beta_star for DD equation (overparametrized)
@qml.qnode(dev_kernel)
def kernel_state(x1):
    """The quantum data encoding."""
    quantum_feature_map(x1)
    return qml.density_matrix(wires=range(n_qubits))

def get_rho_data_matrix(X_data):
    data_size = X_data.shape[0]
    rho_X = [kernel_state(X_data[m, :]).flatten().astype(np.complex128) for m in range(data_size)]
    rho_X = np.array(rho_X, dtype=np.complex128)
    rho_X = rho_X.reshape((data_size, 2**(2*n_qubits)))
    return rho_X

# target beta star and data matrix

n_star = 1000
x_star = generate_evenly_spaced_samples(low-0, high+0, n_star, n_qubits)
y_star = compute_y_from_x_multidim_linear(x_star)

x_star = scaler2.fit_transform(x_star)

X_star = get_rho_data_matrix(x_star)


#%% loop over different number of train data -------------------------------------------------
def get_curves(cutoff, reps):
    subset_sizes = np.arange(24, 104, 5)
    mse_arr_test = [[] for r in range(reps)]
    mse_arr_train = [[] for r in range(reps)]

    for r in range(reps):
        np.random.seed(r)
        for subset_size in subset_sizes:
            (
                X_train,
                X_test,
                y_train,
                y_test,
                indices_train,
                indices_test,
            ) = train_test_split(
                X_star,
                y_star,
                np.arange(X_star.shape[0]),
                random_state=r,
                train_size=subset_size,
                test_size=100,
                shuffle=True,
            )

            U, S, V = np.linalg.svd(X_train)
            min_singular_value = np.min(S[S > 0.0])
            S_cutoff = np.copy(S)
            S_cutoff[S_cutoff < cutoff] = 0.0

            S_inv = np.zeros((V.shape[0], U.shape[0]), dtype=complex)
            for i in range(len(S)):
                if S_cutoff[i] != 0:
                    S_inv[i, i] = 1 / S_cutoff[i]

            beta_hat_cutoff = V.T.conj() @ S_inv @ U.T.conj() @ y_train
            y_train_pred_cutoff = X_train @ beta_hat_cutoff
            train_mse_cutoff = mean_squared_error(y_train, np.real(y_train_pred_cutoff))
            mse_arr_train[r].append(train_mse_cutoff)

            y_test_pred_cutoff = X_test @ beta_hat_cutoff
            test_mse_cutoff = mean_squared_error(y_test, np.real(y_test_pred_cutoff))
            mse_arr_test[r].append(test_mse_cutoff)

    return mse_arr_train, mse_arr_test

singular_value_cutoffs = np.array([10**-3, 3*10**-3, 10**-2, 3*10**-2, 10**-1, 3*10**-1, 1])

# %% Save MSE results for varying number of train samples ------------------------------------
ind = 0
np.savetxt("cutoff_arrange_Synthetic", [np.arange(24, 104, 5)], newline='')
for cutoff in singular_value_cutoffs:
    mse_arr_train, mse_arr_test = get_curves(cutoff, reps)

    mse_te_mean = np.mean(mse_arr_test, axis=0, dtype=np.float64)
    mse_te_std = np.std(mse_arr_test, axis=0) / (np.log(10))

    mse_tr_mean = np.mean(mse_arr_train, axis=0)
    mse_tr_std = np.std(mse_arr_train, axis=0) / (np.log(10))

    np.savetxt(f"cutoff_test_mean_nqubits_{ind}_Synthetic", [mse_te_mean], newline='')
    np.savetxt(f"cutoff_test_std_nqubits_{ind}_Synthetic", [mse_te_std], newline='')

    np.savetxt(f"cutoff_train_mean_nqubits_{ind}_Synthetic", [mse_tr_mean], newline='')
    np.savetxt(f"cutoff_train_std_nqubits_{ind}_Synthetic", [mse_tr_std], newline='')

    ind += 1
