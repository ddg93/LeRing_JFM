#!/usr/bin/env python
# coding: utf-8

"""
Created on Mon Aug 11 14:04:16 2025
@author: davide

Main script to train two-headed LeNet models over datasets of top/side coupled frames labelled with the corresponding object orientation vector n=(n_1,n_2,n_3).

Please cite: "Orientation of flat bodies of revolution in shear flows at low Reynolds number", by D. Di Giusto, L. Bergougnoux & E. Guazzelli, J. of Fluid Mech., 2025
when applying the proposed methodology.


"""

from src.dataset_loader import ImgVec_dataset
from src.twoheaded_CNN import build_two_head_LeNet, make_orientation_loss
from src.tools import plot_losses, plot_errors
from src.read_configuration import configuration
import tensorflow as tf




def main(cnf):
    """
    Main script for training the LeRing 2Headed Lenet model over a synthetic dataset of top/side paired frames
    of an object
    """

    #create the dataset loader
    loader = ImgVec_dataset(particle_folder=cnf.particle_folder)

    #load the dataset
    train_ds = loader.get_dataset(ds=loader.dataset_train,batch_size=cnf.batch_size)
    test_ds = loader.get_dataset(ds=loader.dataset_test,batch_size=256,shuffle=False)

    #create the model
    model = build_two_head_LeNet(conv_filters=cnf.convolutional_filters,conv_kernel_size=cnf.convolutional_kernel_size,
                                 pool_size=cnf.pooling_size, activation=cnf.activation_function, padding=cnf.padding,
                                 dense_units=cnf.dense_units, l2_weight=cnf.l2_regularization_weight, 
                                 dropout_rate=cnf.dropout_rate)

    #create the loss function
    loss_fn = make_orientation_loss(penalty_weight=cnf.penalty_weight)

    #compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(cnf.learning_rate), loss=loss_fn)

    #train the model
    history = model.fit(train_ds, validation_data=test_ds, epochs=cnf.epochs)

    #plot the losses
    plot_losses(history,cnf)

    #plot the errors
    plot_errors(model,train_ds,test_ds,cnf)

    if cnf.save_model == True:
        model.save(cnf.particle_folder+'/model')


if __name__ == "__main__":
    #read the training configuration
    configuration = configuration('configuration.yaml')
    main(configuration)
