# PyTorch Regression from Scratch

This project trains a simple linear regression model in PyTorch to predict concrete compressive strength from concrete mix ingredients and curing age.

The model is implemented manually instead of using `torch.nn.Linear`, so the notebook shows the core pieces of a regression workflow: loading data, preparing inputs and targets, scaling features, computing predictions, calculating loss, running backpropagation, updating parameters, and evaluating the model.

## Files

- `project.ipynb` - Jupyter notebook containing the full data exploration, preprocessing, model training, and evaluation workflow.
- `concrete_data.csv` - Dataset used by the notebook.

## Dataset

The dataset contains 1,030 concrete samples. Each row describes a concrete mix and its measured compressive strength.

Input features:

- `Cement`
- `Blast Furnace Slag`
- `Fly Ash`
- `Water`
- `Superplasticizer`
- `Coarse Aggregate`
- `Fine Aggregate`
- `Age`

Target variable:

- `Strength`

The goal is to predict `Strength` from the eight input features.

## Notebook Workflow

The notebook follows these steps:

1. Import required libraries:
   - `torch`
   - `pandas`
   - `numpy`
   - `sklearn.preprocessing.StandardScaler`
   - `seaborn`
   - `matplotlib`

2. Load the dataset from `concrete_data.csv`.

3. Inspect the dataset:
   - Check the dataset shape.
   - Print column names.
   - Check for missing values.

4. Split the data into:
   - Inputs: all columns except `Strength`
   - Target: the `Strength` column

5. Visualize the data:
   - Histogram of concrete strength
   - Pairplot for feature relationships
   - Violin and box plots for feature distributions

6. Shuffle and split the dataset:
   - 80% training data
   - 20% test data

7. Scale the input features with `StandardScaler`.

8. Convert the NumPy arrays into PyTorch tensors.

9. Initialize model parameters manually:
   - Weight matrix `W`
   - Bias value `b`

10. Define helper functions:
    - `predict(X)` for linear predictions
    - `mse_loss(predictions, targets)` for mean squared error
    - `mae_loss(predictions, targets)` for mean absolute error

11. Train the model with manual gradient descent:
    - Forward pass
    - Loss calculation
    - Backward pass
    - Parameter update
    - Gradient reset


## How to Run

Install the required Python packages:

```bash
pip install torch pandas numpy scikit-learn seaborn matplotlib notebook
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open `project.ipynb` and run the cells from top to bottom.

## Project Purpose

This project is useful for learning how linear regression works under the hood in PyTorch. It avoids high-level model wrappers so the training loop and gradient updates are visible and easy to follow.
