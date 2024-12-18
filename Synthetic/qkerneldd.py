#%%
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from numpy.polynomial import legendre
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.linalg

#%% parameters -------------------------------------------------------------------------------
n_test = 100
n_train_max  = 2500  # this has to be larger than 4**n_qubits
n_qubits = 1
reps = 4

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
x_test = generate_evenly_spaced_samples(low, high, n_test, n_qubits)
y_test = compute_y_from_x_multidim_linear(x_test)

scaler2 = MinMaxScaler(feature_range=(-np.pi/2, np.pi/2))

x_test = scaler2.fit_transform(x_test)

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
        for n in range(n_qubits- 1):
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

@qml.qnode(dev_kernel)
def kernel_state(x1):
    """The quantum data encoding."""
    quantum_feature_map(x1)
    return qml.density_matrix(wires=range(n_qubits))

n_star = 1000

x_star = generate_evenly_spaced_samples(low-0, high+0, n_star, n_qubits)
y_star = compute_y_from_x_multidim_linear(x_star)

x_star = scaler2.transform(x_star)

X_star = [kernel_state(x_star[k, :]).astype(np.complex128) for k in range(n_star)]
X_star = np.array(X_star, dtype=np.complex128)
X_star = X_star.reshape((n_star, 2**(2*n_qubits)))
beta_star = np.linalg.pinv(X_star) @ y_star

X_test = [kernel_state(x_test[k, :]).astype(np.complex128) for k in range(n_test)]
X_test = np.array(X_test, dtype=np.complex128)
X_test = X_test.reshape((n_test, 2**(2*n_qubits)))

y_test_pred = X_test @ beta_star

#%% loop over different number of train data -------------------------------------------------
mse_arr = [[] for r in range(reps)]
mse_arr_theory = [[] for r in range(reps)]
mse_arr_train = [[] for r in range(reps)]
eig_vals_arr = [[] for r in range(reps)]
cond_no_arr = [[] for r in range(reps)]
rank_arr = [[] for r in range(reps)]
params = [[] for r in range(reps)]
x_train_max_arr = [[] for r in range(reps)]
y_train_max_arr = [[] for r in range(reps)]
K_train_arr = []
for r in range(reps):
    print("rep:", r+1, "/", reps)

    x_train_max = np.random.uniform(low=low, high=high, size=(n_train_max, n_qubits))
    y_train_max = compute_y_from_x_multidim_linear(x_train_max)

    x_train_max = scaler2.transform(x_train_max)

    x_train_max_arr[r].append(x_train_max)
    y_train_max_arr[r].append(y_train_max)

    if n_qubits == 1:
        limit_low = 1
        limit_up = 8 
        step = 1
    if n_qubits == 2:
        limit_low = 2**(2*n_qubits)-10  
        limit_up = 2**(2*n_qubits)+11  
        step = 2
    elif n_qubits == 3:
        limit_low = 2**(2*n_qubits)-60  
        limit_up = 2**(2*n_qubits)+61
        step = 20
    elif n_qubits == 4:
        limit_low = 2**(2*n_qubits)-180  
        limit_up = 2**(2*n_qubits)+181
        step = 60
    elif n_qubits == 5:
        limit_low = 2**(2*n_qubits)-720
        limit_up = 2**(2*n_qubits)+721
        step = 240

    m = 0
    for n_train in range(limit_low, limit_up, step):    
        x_train = x_train_max[:n_train]
        y_train = y_train_max[:n_train]
        print(r+1, ": train samples: ", x_train.shape[0], "/", limit_up-1)
     
        
        K_train = kernel_matrix(x_train, x_train).astype(np.float64)
        K_test = kernel_matrix(x_test, x_train).astype(np.float64)
        K_train_arr.append(K_train)

        kernel_ridge_model = KernelRidge(kernel='precomputed', alpha=0.0)
        kernel_ridge_model.fit(K_train, y_train)

        predictions = kernel_ridge_model.predict(K_test)
        train_predictions = kernel_ridge_model.predict(K_train)

        params[r].append(kernel_ridge_model.dual_coef_)

        K_eig_vals = np.linalg.eig(K_train)[0]
        cond_no = np.linalg.cond(K_train)
        rank = np.linalg.matrix_rank(K_train)

        eig_vals_arr[r].append(K_eig_vals)
        cond_no_arr[r].append(cond_no)
        rank_arr[r].append(rank)
        print("Rank of Kernel: ", rank)

        mse = mean_squared_error(np.real(y_test), predictions)
        mse_arr[r].append(mse)
        print("Mean Squared Error Test:", mse)

        train_mse = mean_squared_error(y_train, train_predictions)
        mse_arr_train[r].append(train_mse)
        print("Mean Squared Error Train:", train_mse)

mse_arr = np.array(mse_arr)

# %% plot mse for varying number of train samples -----------------------------------------------------------------------
mse_mean = np.mean(mse_arr, axis=0, dtype=np.float64)
mse_std = np.std(mse_arr, axis=0)  / (np.log(10))

mse_mean_train = np.mean(mse_arr_train, axis=0)
mse_std_train = np.std(mse_arr_train, axis=0)

np.savetxt(f"mse_mean_nqubits_{n_qubits}_Synthetic", [mse_mean], newline='')
np.savetxt(f"mse_std_nqubits_{n_qubits}_Synthetic", [mse_std], newline='')
np.savetxt(f"mse_mean_train_nqubits_{n_qubits}_Synthetic", [mse_mean_train], newline='')
np.savetxt(f"mse_std_train_nqubits_{n_qubits}_Synthetic", [mse_std_train], newline='')
np.savetxt(f"arrange_nqubits_{n_qubits}_Synthetic", [np.arange(limit_low, limit_up, step)], newline='')