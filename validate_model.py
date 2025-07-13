import numpy as np
import utils
import model_predict

def val(model, validation_indices, epoch_count, demo_data, d_x, d_y1, d_y2, time_len=200):

    [_, __, Y1, Y2] = demo_data
    error = 0
    plot_id = np.random.randint(0, len(validation_indices))
    for validation_idx in validation_indices:
        error = 0
        time = np.linspace(0, 1, time_len)
        # permute time
        indices = np.random.permutation(time_len)
        idx = indices[:2]

        time = [time[i] for i in idx]
        y1_condition_points = [[t, Y1[validation_idx, i:i+1]] for t,i in zip(time, idx)]
        y2_condition_points = [[t, Y2[validation_idx, i:i+1]] for t,i in zip(time, idx)]

        means_12, stds_12  = model_predict.predict_12(model, time_len,
                                                          y1_condition_points, d_x, d_y1, d_y2)
        means_11, stds_11 = model_predict.predict_11(model, time_len,
                                                                   y1_condition_points, d_x, d_y1, d_y2)
        means_22, stds_22 = model_predict.predict_22(model, time_len, 
                                                                  y2_condition_points, d_x, d_y1, d_y2)
        means_21, stds_21 = model_predict.predict_21(model, time_len,
                                                          y2_condition_points, d_x, d_y1, d_y2)

        if epoch_count % 2000 == 0 and validation_idx == validation_indices[plot_id]:
            error += utils.validate_model(means_12, stds_12, validation_idx, demo_data, y1_condition_points, epoch_count, forward=False, plot=True)
            error += utils.validate_model(means_11, stds_11, validation_idx, demo_data, y1_condition_points, epoch_count, forward=True, plot=True)
        else:
            error += utils.validate_model(means_12, stds_12, validation_idx, demo_data, y1_condition_points, epoch_count, forward=False, plot=False)
            error += utils.validate_model(means_11, stds_11, validation_idx, demo_data, y1_condition_points, epoch_count, forward=True, plot=False)
            

        error += utils.validate_model(means_22, stds_22, validation_idx, demo_data, y2_condition_points, epoch_count, forward=False, plot=False)
        error += utils.validate_model(means_21, stds_21, validation_idx, demo_data, y2_condition_points, epoch_count, forward=True, plot=False)
    
    return error / (len(validation_indices) * 4)  # average error over all validation indices and both forward and inverse predictions
