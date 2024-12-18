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
n_train_max  = 1000
n_qubits = 3
alpha_list = [0.0, 0.0001, 0.001, 0.01, 0.1]

#%% different functions to generate data ---------------------------------------------------------------

def generate_evenly_spaced_samples(low, high, n_test, n_qubits):
    num_points_per_dim = int(np.ceil(n_test ** (1 / n_qubits)))
    linspaces = [np.linspace(start=low, stop=high, num=num_points_per_dim) for _ in range(n_qubits)]
    grid = np.meshgrid(*linspaces, indexing='ij')
    x_test_full = np.stack(grid, axis=-1).reshape(-1, n_qubits)
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
    y_data = 2*x_data + np.cos(15*x_data) + np.random.normal(0, noise_level, n_samples).astype(np.float64)
    return x_data.reshape(-1, 1), y_data

def compute_y_from_x(X: np.ndarray):
    return np.add(2.0 * X, np.cos(X * 25))[:, 0]

def generate_multidim_linear_cos_data(n_samples, noise_level=0.001):
    x_data = np.random.uniform(-1, 1, size=(n_samples, n_qubits)).astype(np.float64)
    weights_lin =  np.zeros(n_qubits)
    weights = np.arange(1, n_qubits + 1, dtype=np.float64) / n_qubits
    y_data = np.dot(x_data, weights_lin) + np.cos(3*np.dot(x_data, weights)) + np.random.normal(0, noise_level, n_samples).astype(np.float64)
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

if x_test.shape[1] > 1:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_test[:, 0], x_test[:, 1], y_test, label="test data")
    plt.legend()
    plt.show()
else:    
    plt.scatter(x_test, y_test, label="test data")
    plt.legend()
    plt.show()

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

# compute beta_star for DD equation (overparametrized) 
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
reps = len(alpha_list)
mse_arr = [[] for r in range(reps)]
mse_arr_theory = [[] for r in range(reps)]
mse_arr_train = [[] for r in range(reps)]
eig_vals_arr = [[] for r in range(reps)]
cond_no_arr = [[] for r in range(reps)]
rank_arr = [[] for r in range(reps)]
params = [[] for r in range(reps)]
x_train_max_arr = [[] for r in range(reps)]
y_train_max_arr = [[] for r in range(reps)]
mse_arr_alpha = []  # Stores MSE for all alphas and runs
std_arr_alpha = []  # Stores std deviation for all alphas
r = 0
for alpha in alpha_list:
    print("reg. parameter:", alpha)
    mse_runs = []  # Collect MSE for all 5 runs for this alpha

    np.random.seed(11)
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
        limit_low = 24  
        limit_up = 104
        step = 5
    elif n_qubits == 4:
        limit_low = 2**(2*n_qubits)-180  
        limit_up = 2**(2*n_qubits)+181
        step = 60

    for run in range(5):  # Loop for 5 runs
        print(f"Run {run + 1} for alpha {alpha}")
        np.random.seed(11 + run)  # Change the seed for each run
        
        x_train_max = np.random.uniform(low=low, high=high, size=(n_train_max, n_qubits))
        y_train_max = compute_y_from_x_multidim_linear(x_train_max)

        x_train_max = scaler2.transform(x_train_max)

        mse_per_n_train = []  # Store MSE for each n_train in the current run

        for n_train in range(limit_low, limit_up, step):    
            x_train = x_train_max[:n_train]
            y_train = y_train_max[:n_train]

            K_train = kernel_matrix(x_train, x_train).astype(np.float64)
            K_test = kernel_matrix(x_test, x_train).astype(np.float64)

            kernel_ridge_model = KernelRidge(kernel='precomputed', alpha=alpha)
            kernel_ridge_model.fit(K_train, y_train)

            predictions = kernel_ridge_model.predict(K_test)

            mse = mean_squared_error(np.real(y_test), predictions)
            mse_per_n_train.append(mse)

        mse_runs.append(mse_per_n_train)  # Collect mean MSE for this run

    mse_runs = np.array(mse_runs)
    print(mse_runs)
    mse_mean = np.mean(mse_runs, axis=0)
    print(mse_mean)
    mse_std = np.std(mse_runs, axis=0) / (np.log(10))
    
    np.savetxt(f"regularization_mse_mean_{r}", [mse_mean], newline='')
    np.savetxt(f"regularization_mse_std_{r}", [mse_std], newline='')

    mse_arr_alpha.append(mse_mean)
    std_arr_alpha.append(mse_std)

    print(f"Alpha {alpha}: Mean MSE = {mse_mean}, Std Dev = {mse_std}")

    r+=1

mse_arr = np.array(mse_arr)

np.savetxt("regularization_arrange_Synthetic", [np.arange(limit_low, limit_up, step)], newline='')