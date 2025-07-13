from cProfile import label
from fileinput import filename
from re import A, X
from turtle import color
from pyparsing import lineEnd
from sympy import li
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import model

def validate_model(means, stds, idx, demo_data, condition_points, epoch_count, forward, plot=False):

    _, __, Y1, Y2= demo_data

    if forward:
        target_demo = Y1[idx,:,:]
    else:
        target_demo = Y2[idx,:,:]

    error = torch.mean(torch.nn.functional.mse_loss(means[:,:], target_demo[:,:]))

    if plot:
        plot_test(idx, Y1, Y2, means, stds, condition_points, epoch_count)
    
    return error

def plot_test(idx, Y1, Y2, means, stds, condition_points, epoch_count):
    d_N = Y1.shape[0]
    num_dim = Y1.shape[2]
    T_forward = np.linspace(0,1,Y1.shape[1])
    T_inverse = np.linspace(0,1,Y2.shape[1])

    ## plot forward and inverse trajectories for each dimension, add subplots, 4 above, 3 below

    plt.figure(figsize=(15, 15))
    axes = []
    for i in range(num_dim):
        axes.append(plt.subplot(4, 3, i+1))

    ax = [[axes[0]]]#], ax2, ax3], [ax4, ax5, ax6], [ax7]]#, ax8]]
    dim_plot_dict = {0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2), 6: (2,0), 7: (2,1)}

    for dim in range(num_dim):
        plot_idx = dim_plot_dict[dim]
        ax[plot_idx[0]][plot_idx[1]].set_title(f"Joint {dim}")
        for j in range(d_N):
            if j == idx:
                ax[plot_idx[0]][plot_idx[1]].plot(T_forward, Y1[j,:,dim], color='blue', label='Forward', alpha=0.5)
                ax[plot_idx[0]][plot_idx[1]].plot(T_inverse, Y2[j,:,dim], color='red', label='Expected (Inverse)', alpha=0.5)
                continue
            ax[plot_idx[0]][plot_idx[1]].plot(T_forward, Y1[j,:,dim], color='black', alpha=0.1)
            ax[plot_idx[0]][plot_idx[1]].plot(T_inverse, Y2[j,:,dim], color='black', alpha=0.1)

        ax[plot_idx[0]][plot_idx[1]].plot(T_forward, means[:,dim].detach().numpy(), color='green', label='Prediction')
        ax[plot_idx[0]][plot_idx[1]].errorbar(T_forward, means[:,dim].detach().numpy(), yerr=stds[:,dim].detach().numpy(), color='black', alpha=0.2)
        
        for i in range(len(condition_points)):
            cd_pt_x = condition_points[i][0]
            cd_pt_y = condition_points[i][1][0][dim]
            if i == 0:
                pass
                ax[plot_idx[0]][plot_idx[1]].scatter(cd_pt_x, cd_pt_y, color='black', label='Observations')
                continue
            ax[plot_idx[0]][plot_idx[1]].scatter(cd_pt_x, cd_pt_y, color='black')

    plt.suptitle(f"Prediction for epoch {epoch_count}")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def plot_results(best_mean, best_std, Y1, Y2, idx, condition_points, errors, losses, time_len, d_N, plot_errors=True, test_dist=None):
    
    num_dim = 7

    T = np.linspace(0,1,time_len)

    plt.figure(figsize=(15, 15))
    ax1 = plt.subplot(4, 3, 1)
    ax2 = plt.subplot(4, 3, 2)
    ax3 = plt.subplot(4, 3, 3)
    ax4 = plt.subplot(4, 3, 4)
    ax5 = plt.subplot(4, 3, 5)
    ax6 = plt.subplot(4, 3, 6)
    ax7 = plt.subplot(4, 3, 7)
    ax = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7]]
    dim_plot_dict = {0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2), 6: (2,0), 7: (2,1)}

    for dim in range(num_dim):
        plot_idx = dim_plot_dict[dim]
        ax[plot_idx[0]][plot_idx[1]].set_title(f"Joint {dim}", fontsize=20)
        for j in range(d_N):
            if j == 0: 
                #ax[plot_idx[0]][plot_idx[1]].plot(T, Y1[j,:,dim], color='green', alpha=0.1, label='Forward Trajectories (Green)')
                ax[plot_idx[0]][plot_idx[1]].plot(T, Y2[j,:,dim], color='blue', alpha=0.1, label='Inverse Trajectories')
                if j == idx:
                    ax[plot_idx[0]][plot_idx[1]].plot(T, Y2[j,:,dim], color='blue', alpha=0.1, label='Ground Truth')
                continue
            if j == idx:
                ax[plot_idx[0]][plot_idx[1]].plot(T, Y2[j,:,dim], color='blue', alpha=0.1)
            #ax[plot_idx[0]][plot_idx[1]].plot(T, Y1[j,:,dim], color='green', alpha=0.1)
            ax[plot_idx[0]][plot_idx[1]].plot(T, Y2[j,:,dim], color='blue', alpha=0.1)

        ax[plot_idx[0]][plot_idx[1]].plot(T, best_mean[:,dim].detach().numpy(), color='black', label='Prediction')
        ax[plot_idx[0]][plot_idx[1]].errorbar(T, best_mean[:,dim].detach().numpy(), yerr=best_std[:,dim].detach().numpy(), color='black', alpha=0.2)
        
        for i in range(len(condition_points)):
            cd_pt_x = condition_points[i][0]
            cd_pt_y = condition_points[i][1][dim]
            if i == 0:
                pass
                ax[plot_idx[0]][plot_idx[1]].scatter(cd_pt_x, cd_pt_y, color='black', label='Observations')
                continue
            ax[plot_idx[0]][plot_idx[1]].scatter(cd_pt_x, cd_pt_y, color='black')

    for ax_ in ax[0]:
        ax_.grid(alpha=0.3)
        
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.savefig(f"../figs/Results.png")
    plt.show()

############################################################################################################

"""
def plot_latent_space(model, observations_f, observations_i):
    with torch.no_grad():
        l_f = model.encoder1(observations_f) # condition points is (n, d_x + d_y), l is (n, 128)
        l_i = model.encoder2(observations_i)
    l_f = np.array(l_f)
    l_i = np.array(l_i)
    l = np.concat((l_f,l_i), axis=0)

    
    for i in range(len(l_f)):
        plt.scatter(i,np.linalg.norm(l_f[i]-l_i[i]), color='blue')
    plt.show()

    print(l.shape)
    l = l.squeeze(1)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(l)
    return pca_result


def tsne_analysis(model, observations_f, observations_i):
    with torch.no_grad():
        l_f = model.encoder1(observations_f) # condition points is (n, d_x + d_y), l is (n, 128)
        l_i = model.encoder2(observations_i)
    l_f = np.array(l_f)
    l_i = np.array(l_i)
    l = np.concat((l_f,l_i), axis=0)

    l = l.squeeze(1)

    tsne = TSNE(n_components=2, random_state=41)
    tsne_result = tsne.fit_transform(l)
    return tsne_result
"""