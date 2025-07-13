import torch
import torch.nn as nn
import loss_utils

class DualEncoderDecoder(nn.Module):
    def __init__(self, d_x, d_y1, d_y2,
                 encoder1_hidden_dims=[128, 128],
                 encoder2_hidden_dims=[128, 128],
                 decoder1_hidden_dims=[128, 128],
                 decoder2_hidden_dims=[128, 128]):
        """
        Initializes the DualEncoderDecoder module.

        Args:
            d_x (int): Dimension of time stamp (Usually 1).
            d_y1 (int): Dimension of the first modality.
            d_y2 (int): Dimension of the second modality.
            learned_param_dim (int): Dimension of the learned parameter.
            encoder1_hidden_dims (list, optional): List of hidden layer dimensions for the first encoder. Defaults to [128, 128, 128].
            encoder2_hidden_dims (list, optional): List of hidden layer dimensions for the second encoder. Defaults to [128, 128, 128].
            decoder1_hidden_dims (list, optional): List of hidden layer dimensions for the first decoder. Defaults to [128, 128, 128].
            decoder2_hidden_dims (list, optional): List of hidden layer dimensions for the second decoder. Defaults to [128, 128, 128].
        """
        super(DualEncoderDecoder, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        self.d_y2 = d_y2

        # Encoders
        final_encoder_dim = 256  # As per the original architecture
        enc1_full_dims = [d_x + d_y1] + encoder1_hidden_dims + [final_encoder_dim]
        enc2_full_dims = [d_x + d_y2] + encoder2_hidden_dims + [final_encoder_dim]

        # Decoders
        dec_input_dim = d_x + final_encoder_dim
        dec1_full_dims = [dec_input_dim] + decoder1_hidden_dims + [d_y1 * 2]
        dec2_full_dims = [dec_input_dim] + decoder2_hidden_dims + [d_y2 * 2]

        # --- Build the sequential models ---
        self.encoder1 = self._build_sequential(enc1_full_dims)
        self.encoder2 = self._build_sequential(enc2_full_dims)
        self.decoder1 = self._build_sequential(dec1_full_dims)
        self.decoder2 = self._build_sequential(dec2_full_dims)

    def _build_sequential(self, dims):
        """Builds a sequential model from a list of layer dimensions."""
        layers = []
        # Loop through all but the last two dimensions to create hidden layers
        for i in range(len(dims) - 2):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.ReLU()
            ])
        # Add the final linear layer without normalization or activation
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def forward(self, obs, mask, x_tar, MODE=0):
        # obs: (num_traj, max_obs_num, 2*d_x + d_y1 + d_y2) 
        # mask: (num_traj, max_obs_num, 1)
        # x_tar: (num_traj, num_tar, d_x)

        mask_forward, mask_inverse = mask[0], mask[1] # (num_traj, max_obs_num, max_obs_num)

        obs_f = obs[:, :, :self.d_x+self.d_y1]  # (num_traj, max_obs_num, d_x + d_y1)
        obs_i = obs[:, :, self.d_x+self.d_y1:2*self.d_x+self.d_y1+self.d_y2]  # (num_traj, max_obs_num, d_x + d_y2)

    
        r1 = self.encoder1(obs_f)  # (num_traj, max_obs_num, H)

        # Instead of averaging over the observations, we use the mask to compute the mean
        masked_r1 = torch.bmm(mask_forward, r1) # (num_traj, max_obs_num, H) 
        sum_masked_r1 = torch.sum(masked_r1, dim=1) # (num_traj, H)
        L_1 = sum_masked_r1 / (torch.sum(mask_forward, dim=[1,2]).reshape(-1,1) + 1e-10)

        # Repeat L_F for multiple target points.
        # During training, the number of target points is 1, but during inference, it can be more.
        L_1 = L_1.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (num_traj, num_tar, H)

        r2 = self.encoder2(obs_i)  # (num_traj, max_obs_num, H)
        masked_r2 = torch.bmm(mask_inverse, r2)
        sum_masked_r2 = torch.sum(masked_r2, dim=1) # (num_traj, H)
        L_2 = sum_masked_r2 / (torch.sum(mask_inverse, dim=[1,2]).reshape(-1,1) + 1e-10)
        L_2 = L_2.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (num_traj, num_tar, H)

        latent = torch.zeros(0)
        if MODE == 0: # Training mode
            p1 = torch.rand(1)
            p2 = torch.rand(1)
            p1 = p1 / (p1 + p2)
            latent = L_1 * p1 + L_2 * (1-p1)  # (num_traj, num_tar, H)
        elif MODE == 1: # Prediction from observation from modality 1
            latent = L_1 
        elif MODE == 2: # Prediction from observation from modality 2
            latent = L_2

        concat = torch.cat((latent, x_tar), dim=-1)  # (num_traj, num_tar, H + d_x) 
        
        output1 = self.decoder1(concat)  # (num_traj, num_tar, 2*d_y1)
        output2 = self.decoder2(concat)  # (num_traj, num_tar, 2*d_y2)

        return torch.cat((output1, output2), dim=-1)
    
def get_training_sample(validation_indices, demo_data, 
                        OBS_MAX, d_N, d_x, d_y1, d_y2, time_len, batch_size):

    X1, X2, Y1, Y2 = demo_data

    traj_multinom = torch.ones(d_N) # multinomial distribution for trajectories

    # Uncomment the following lines to remove validation indices from the training set
    #for i in validation_indices:
    #    traj_multinom[i] = 0
    
    traj_indices = torch.multinomial(traj_multinom, batch_size, replacement=False) # random indices of trajectories

    obs_num_list = torch.randint(0, OBS_MAX, (2*batch_size,)) + 1  # random number of obs. points
    max_obs_num = OBS_MAX
    observations = torch.zeros((batch_size, max_obs_num, 2*d_x + d_y1 + d_y2))
    
    # Create masks to indicate which observations are valid.
    # This enables the model to handle variable-length sequences inside the batch.
    mask_y1 = torch.zeros((batch_size, max_obs_num, max_obs_num))
    mask_y2 = torch.zeros((batch_size, max_obs_num, max_obs_num))

    target_X = torch.zeros((batch_size, 1, d_x))
    target_Y1 = torch.zeros((batch_size, 1, d_y1))
    target_Y2 = torch.zeros((batch_size, 1, d_y2))

    T = torch.ones(time_len)
    for i in range(batch_size):
        
        traj_index = int(traj_indices[i])
        obs_num_y1 = int(obs_num_list[i])
        obs_num_y2 = int(obs_num_list[batch_size + i])
        obs_indices_y1 = torch.multinomial(T, obs_num_y1, replacement=False)
        obs_indices_y2 = torch.multinomial(T, obs_num_y2, replacement=False)

        observations[i][:obs_num_y1, :d_x] = X1[0][obs_indices_y1] 
        observations[i][:obs_num_y1, d_x:d_x+d_y1] = Y1[traj_index][obs_indices_y1] 
        observations[i][:obs_num_y2, d_x + d_y1:2*d_x + d_y1] = X2[0][obs_indices_y2] 
        observations[i][:obs_num_y2, 2*d_x + d_y1:] = Y2[traj_index][obs_indices_y2]

        mask_y1[i][:obs_num_y1, :obs_num_y1] = 1
        mask_y2[i][:obs_num_y2, :obs_num_y2] = 1

        target_index = torch.multinomial(T, 1)
        target_X[i] = X1[0][target_index]
        target_Y1[i] = Y1[traj_index][target_index]
        target_Y2[i] = Y2[traj_index][target_index]
        
    return observations, [mask_y1, mask_y2], target_X, target_Y1, target_Y2
    
def loss(output, target_y1, target_y2, d_y1, d_y2):
    
    log_prob = loss_utils.log_prob_loss(output, target_y1, target_y2, d_y1, d_y2)

    return log_prob
