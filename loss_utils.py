import torch
import torch.nn.functional as F
import torch.distributions as D

def log_prob_loss(output, targets_y1, targets_y2, d_y1, d_y2): # output (num_traj, num_tar, 2*d_y1 + 2*d_y2), targets (num_traj, num_tar, d_y1 + d_y2)

    means_y1, stds_y1, means_y2, stds_y2 = output[:, :, :d_y1], output[:, :, d_y1:2*d_y1], +\
                output[:, :, 2*d_y1:2*d_y1+d_y2], output[:, :, 2*d_y1+d_y2:] # (num_traj, num_tar, d_y1 + d_y2)

    stds_y1 = F.softplus(stds_y1)
    normal_f = D.Normal(means_y1, stds_y1)
    log_prob_y1 = normal_f.log_prob(targets_y1) 
    
    stds_y2 = F.softplus(stds_y2) # (num_traj, num_tar, d_y1)
    normal_i = D.Normal(means_y2, stds_y2)
    log_prob_y2 = normal_i.log_prob(targets_y2) 

    total_loss = log_prob_y1 + log_prob_y2
    return -0.5 * total_loss.mean()  # scalar