# Double descent in quantum machine learning

This repository contains the data and code in the paper "Double descent in quantum machine learning," available on [arXiv](arXivURL). The code relies on the PennyLane [(GitHub)](https://github.com/PennyLaneAI/pennylane) package. Please ensure that this package is installed before running the code.


## Repository structure
The repository is organized into folders corresponding to the three different experiments described in the manuscript.

#### Empirical evidence of double descent in QML (Section V)
The code and data for these experiments are located in the following folders:
- `Synthetic`
- `MNIST_Fashion`
- `Housing`
  
Each folder contains a Python script that can be executed to run experiments on the respective dataset. Key parameters, including the number of qubits (`n_qubits`), are defined at the beginning of each script. To reproduce the experiments, adjust `n_qubits` as follows:
- Set `n_qubits` to values between 2 and 4 for standard runs.
- For the more computationally demanding 5-qubit experiments, use the dedicated subfolder to each experimental repetition in parallel.

#### Ablation experiments (Section VI)
The ablation experiments are found in the `ablation` folder, which contains three subfolders:
- `cutoff`
- `leading`
- `residual`

Each subfolder includes a Python script to execute the corresponding ablation experiment.

#### The case of projected quantum kernels (Section VII)
The projected quantum kernel experiments are stored in the `projected_kernel` folder. This folder is further divided into three subfolders corresponding to the datasets:
- `Synthetic`
- `MNIST_Fashion`
- `Housing`

Each dataset folder contains a Python script for running the experiments. As with the first set of experiments, you can adjust the number of qubits (n_qubits) at the beginning of each script. Supported values range from 2 to 5.


## Running the code

To run the code, follow these steps:

1. Install the required packages: PennyLane.

2. Choose the appropriate code file based on the experiments you want to run.

3. Set the desired values for the argument `n_qubits` (if applicable) to customize the execution.

4. Execute the chosen code file, ensuring the required packages are accessible.

For further assistance or inquiries, please refer to the paper or contact the authors directly.
