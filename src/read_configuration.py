#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:04:16 2025
@author: davide

Configuration class template
"""


import yaml

class configuration:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Dataset settings
        dataset_cfg = cfg.get('dataset', {})
        self.particle_folder = dataset_cfg['particle_folder']
        self.image_size = tuple(dataset_cfg['image_size'])
        self.file_format = dataset_cfg['file_format']
        self.test_size = dataset_cfg['test_size']
        self.random_state = dataset_cfg['random_state']
        self.batch_size = dataset_cfg['batch_size']
        
        # Model settings
        model_cfg = cfg.get('model', {})
        self.convolutional_filters = model_cfg['convolutional_filters']
        self.convolutional_kernel_size = model_cfg['convolutional_kernel_size']
        self.pooling_size = model_cfg['pooling_size']
        self.activation_function = model_cfg['activation_function']
        self.padding = model_cfg['padding']
        self.dense_units = model_cfg['dense_units']
        self.l2_regularization_weight = model_cfg['l2_regularization_weight']
        self.dropout_rate = model_cfg['dropout_rate']  # will be None if null in YAML
        self.save_model = model_cfg['save_model']
        
        # Training settings
        training_cfg = cfg.get('training', {})
        self.learning_rate = training_cfg['learning_rate']
        self.penalty_weight = training_cfg['penalty_weight']
        self.epochs = training_cfg['epochs']
        self.save_plots = training_cfg['save_plots']

