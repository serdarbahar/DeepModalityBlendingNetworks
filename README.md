# 2 Modality Implementation of Deep Modality Blending Networks

## Scripts

### dual_enc_dec_cnmp.py
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
