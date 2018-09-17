# Leave Times

## Overview

This project contains a basic python application for predicting bus leave times with different machine learning models (linear regression, DNN regression).

The project is highly modular, a different model can be used with the same data and training process by inheriting from the `Model` class and creating a new instance of it in `leave_times/__init__.py` (see [included models](#project-structure) as examples).

## Requirements

- Python 3 (developed using v3.5.4)

## Setting up

Make sure `python`, `pip` and the `Scripts` directory are in the path and run the following commands:

  1. `pip install --upgrade --user virtualenv`
  2. `cd {project directory}`
  3. `virtualenv env`
  4. `env\Scripts\activate`
  5. `pip install pip-tools`

### For development

  6. `pip-sync requirements-dev.txt`

### For production

  6. `pip-sync requirements.txt`

## Start application

Navigate to the `leave_times` subfolder using command line, then run `python __init__.py`.

## Project structure

- Models
  - Linear regression: `leave_times/linear_regression.py`
  - Neural Network: `leave_times/neural_network.py`
- Training data: `data/log.csv`
- Test data: `data/log_unlabeled.csv`
