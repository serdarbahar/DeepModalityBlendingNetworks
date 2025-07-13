# 2 Modality Implementation of Deep Modality Blending Networks

## Scripts

### model.py
The model, training sample construction logic, and loss function.

### loss_utils.py
The loss criterions. Currently, only negative log-likelihood loss is used.

### model_predict.py
Helper functions for trajectory generation. For example, predict_12(.) describes the trajectory generated of the 2nd modality given observation from 1st modality

### train.ipynb
Jupyter notebook file to load data and train the model.

### utils.py
Helper functions to visualize model capabilities during or after training.

### validate_model.py
Validation logic for the training. It uses a different criterion (MSE) for 4 possible channels for trajectory generation for trajectory indices in the validation_indices list.

## Notes
- Time-series data for the two modalities is assumed to have equal length, taken with a uniform sampling rate. Data is then time-stamped to the interval between 0-1 with 200 timesteps
- The data in the same indices for two modalities (e.g. Y1[7] and Y2[7]) are assumed to be paired and the model learns a joint representation of the pair. For example, every index can contain different modalities for certain environments, etc.
- The training is not done on GPU because of specific sampling operations during training (i.e. model.py/get_training_sample() function)
- Training usually takes orders of ten/hundred thousand epochs.
