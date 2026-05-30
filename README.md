# PyTorch Regression from Scratch

This project trains a simple linear regression model in PyTorch to predict concrete compressive strength from concrete mix ingredients and curing age.

The model is implemented manually instead of using `torch.nn.Linear`, so the notebook shows the core pieces of a regression workflow: loading data, preparing inputs and targets, scaling features, computing predictions, calculating loss, running backpropagation, updating parameters, and evaluating the model.

## Files

- `project-linear.ipynb` 
- `project-non_linear.ipynb` 
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

## Project Purpose

This project is useful for learning how linear regression works under the hood in PyTorch. It avoids high-level model wrappers so the training loop and gradient updates are visible and easy to follow.
