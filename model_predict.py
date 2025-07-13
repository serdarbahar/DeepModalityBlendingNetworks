import torch
import numpy as np

def predict_12(model, time_len, condition_points, d_x, d_y1, d_y2):
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y1))
    mock_obs = torch.zeros((1, num_conditions, d_x + d_y2))
    mask = torch.eye(num_conditions).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0]).reshape(1,1)
        y_obs = condition[1].reshape(1,d_y1)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 
    
    means = torch.zeros(0)
    stds = torch.zeros(0)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len).reshape(1, time_len, -1)
        obs = torch.cat((obs, mock_obs), dim=-1)
        output = model(obs, mask, T, MODE=1)
        _, __, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2))
        means = mean2
        stds = std2
    
    return means[0], stds[0]

def predict_11(model, time_len, condition_points, d_x, d_y1, d_y2):
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y1))
    mock_obs = torch.zeros((1, num_conditions, d_x + d_y2))
    mask = torch.eye(num_conditions).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0]).reshape(1,1)
        y_obs = condition[1].reshape(1, d_y1)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 
    
    means = torch.zeros(0)
    stds = torch.zeros(0)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len).reshape(1, time_len, -1)
        obs = torch.cat((obs, mock_obs), dim=-1)
        output = model(obs, mask, T, MODE=1)
        mean1, std1, _, __ = output.chunk(4, dim=-1)
        std1 = np.log(1+np.exp(std1))
        means = mean1
        stds = std1
    
    return means[0], stds[0]


def predict_21(model, time_len, condition_points, d_x, d_y1, d_y2):
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y2))
    mock_obs = torch.zeros((1, num_conditions, d_x + d_y1))
    mask = torch.eye(num_conditions).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0]).reshape(1,1)
        y_obs = condition[1].reshape(1, d_y2)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 

    means = torch.zeros(0)
    stds = torch.zeros(0)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len).reshape(1, time_len, -1)
        obs = torch.cat((mock_obs, obs), dim=-1)
        output = model(obs, mask, T, MODE=2)
        mean1, std1, _, __ = output.chunk(4, dim=-1)
        std1 = np.log(1+np.exp(std1))
        means = mean1
        stds = std1
    
    return means[0], stds[0]

def predict_22(model, time_len, condition_points, d_x, d_y1, d_y2):
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y2))
    mock_obs = torch.zeros((1, num_conditions, d_x + d_y1))
    mask = torch.eye(num_conditions).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0]).reshape(1,1)
        y_obs = condition[1].reshape(1, d_y2)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 

    means = torch.zeros(0)
    stds = torch.zeros(0)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len).reshape(1, time_len, -1)
        obs = torch.cat((mock_obs, obs), dim=-1)
        output= model(obs, mask, T, MODE=2)
        _, __, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2))
        means = mean2
        stds = std2
    
    return means[0], stds[0]