#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:04:16 2025
@author: davide

Contains:
	- plot_losses to display the training and testing losses against the epochs
	- plot_errors to calculate and display training and testing errors against true values
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_losses(history,cnf):
    """plot train and test losses against the epochs"""
    plt.figure(figsize=(3,2))
    plt.plot(history.epoch,history.history['loss'],label=r'$Train$')
    plt.plot(history.epoch,history.history['val_loss'],label=r'$Train$')
    plt.legend()
    plt.yscale('log');
    plt.xlabel(r'$epochs$')
    plt.ylabel(r'$Losses$');
    if cnf.save_plots:
        plt.savefig(cnf.particle_folder+'/losses.png', transparent=True, dpi=96, format='png', bbox_inches='tight')

def plot_errors(model,train_ds,test_ds,cnf):
    """Calculate train and test errors and plot histogram distributions"""
    #forecast on the train and test datasets
    forecast_train = model.predict(train_ds)
    forecast_test = model.predict(test_ds)
    #get the true values
    train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    test_labels = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    #calculate the L2-norm errors
    norm_train = np.linalg.norm(forecast_train-train_labels,axis=-1)
    norm_test = np.linalg.norm(forecast_test-test_labels,axis=-1)

    #plot the errors
    fig,axs = plt.subplots(1,2,figsize=(6,2))
    axs[0].hist(np.linalg.norm(forecast_train-train_labels,axis=-1),bins=50,label=r'$Train$');

    axs[0].hist(np.linalg.norm(forecast_test-test_labels,axis=-1),bins=50,label=r'$Test$');
    axs[0].legend();
    axs[0].set_xlim(0,1)
    axs[0].set_xlabel(r'$\vert\vert \vec{n}_{true}-\vec{n}_{pred}\vert\vert_2$')
    axs[0].set_ylabel(r'$Occurences$');

    axs[1].hist(abs(forecast_train[:,0]-train_labels[:,0]),label=r'$n_1$');
    axs[1].hist(abs(forecast_train[:,1]-train_labels[:,1]),label=r'$n_2$');
    axs[1].hist(abs(forecast_train[:,2]-train_labels[:,2]),label=r'$n_3$');
    axs[1].set_xlabel(r'$\vert {n}_{i,true} - {n}_{i,pred}\vert$')
    axs[1].legend();
    if cnf.save_plots:
        plt.savefig(cnf.particle_folder+'/errors.png', transparent=True, dpi=96, format='png', bbox_inches='tight')

