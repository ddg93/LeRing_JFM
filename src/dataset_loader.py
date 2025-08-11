#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:15:16 2025
@author: davide

Containts the ImgVec_dataset class intended to load a perpendicular frames dataset labelled with the represented object orientation vector n.
Given a particle_folder, the script handles images stored inside particle_folder/dataset/{top_view,side_view}/
Particle orientation vector labels are obtained from the image names.

"""

import tensorflow as tf
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

class ImgVec_dataset:
    def __init__(self, particle_folder, image_size=(100, 100), file_format='.jpg', test_size=0.2, random_state=42):
        """Initialize the particle_folder, image size and images file format and build the dataset loader"""
        self.home = particle_folder
        self.dataset_dir = os.path.join(particle_folder, 'dataset')
        self.top_dir = os.path.join(self.dataset_dir, 'top_view')
        self.side_dir = os.path.join(self.dataset_dir, 'side_view')
        self.image_size = image_size
        self.format = file_format
        self.test_size = test_size
        self.random_state = random_state
	#list files and split
        self.list_and_split()
        
	#build the training dataset
        self.dataset_train = self.build_dataset(self.top_train,self.side_train)
	#build the testing dataset
        self.dataset_test  = self.build_dataset(self.top_test,self.side_test)

    @staticmethod
    def parse_label_from_filename(filename):
        """Extract particle orientation vector from filename string"""
        #convert to string
        filename = filename.numpy().decode("utf-8")
        #find the floats
        res = re.findall(r"[-+]?(?:\d*\.*\d+)", filename)
        #last six are: n1,value,n2,value,n3,value -> select the three values 
        vec = [float(res[-5]), float(res[-3]), float(res[-1])]
        return np.array(vec, dtype=np.float32)
    
    def load_and_preprocess(self,path):
        """load the path image, resize if necessary and normalize"""
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, self.image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def load_image_paths(self, folder):
        """load the all the image labels inside folder and sort"""
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(self.format)]
        return sorted(files)

    def list_and_split(self):
        """list all files in the top and side folders, check files match and split into training and testing file lists"""
        #Get sorted file lists
        top_files = self.load_image_paths(self.top_dir)
        side_files = self.load_image_paths(self.side_dir)

        #safety check: assert that labels match between side and top files
        for t, s in zip(top_files, side_files):
            if os.path.basename(t)[3:] != os.path.basename(s)[4:]:
                raise ValueError(f"Label mismatch: {t} vs {s}")

        #split into train and test files
        self.side_train, self.side_test, self.top_train, self.top_test = train_test_split(side_files, top_files, test_size=self.test_size, random_state=self.random_state)

    def build_dataset(self,top_files,side_files):
        """build the dataset with (x=(side-top), y=vector) paired images and particle orientation vector label from the given lists of top and side files"""
        #create the built-in tf datasets from top and side file lists
        top_ds = tf.data.Dataset.from_tensor_slices(top_files)
        side_ds = tf.data.Dataset.from_tensor_slices(side_files)
        #apply the load and preprocess mapping, allow for parallelism
        top_ds  = top_ds.map(self.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        side_ds = side_ds.map(self.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        #get the label strings and apply python function to get the vectors
        label_ds = tf.data.Dataset.from_tensor_slices(top_files)
        label_ds = label_ds.map(lambda x: tf.py_function(func=self.parse_label_from_filename, inp=[x], 
                                                         Tout=tf.float32),num_parallel_calls=tf.data.AUTOTUNE)

        #assemble the top and side datasets to generate the final one with format: ((side, top), label)
        dataset = tf.data.Dataset.zip(((side_ds, top_ds), label_ds))
        #return the loader
        return dataset

    def get_dataset(self, ds, batch_size=32, shuffle=True):
        """Apply the dataset loader with batch_size and optional shuffling"""
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

