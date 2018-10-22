"""
Reimplementing segan paper as close as possible. 
Deepak Baby, UGent, June 2018.
"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, flatten, fully_connected
import numpy as np
import keras
from keras.layers import Input, Dense, Conv1D, Conv2DTranspose, BatchNormalization
from keras.layers import LeakyReLU, PReLU, Reshape, Concatenate, Flatten, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from normalizations import InstanceNormalization
#from bnorm import VBN
#Conv2DTranspose = tf.keras.layers.Conv2DTranspose
keras_backend = tf.keras.backend
keras_initializers = tf.keras.initializers
from data_ops import *

import h5py

def generator(opts):
    kwidth=opts['filterlength']
    strides= opts['strides']
    pool = strides
    g_enc_numkernels = opts ['g_enc_numkernels']
    g_dec_numkernels = opts ['g_dec_numkernels']
    window_length = opts['window_length']
    featdim = opts ['featdim']
    batch_size = opts['batch_size']
    if opts['GT_init_G']:
        gt = np.expand_dims(opts['gt'], axis=1)
        num_gt_filters = gt.shape[2]
        gt_filterlength = gt.shape[0]
        gt_bias = np.zeros((num_gt_filters,))
        
    use_bias = True
    skips = []
    #kernel_init = keras.initializers.TruncatedNormal(stddev=0.02)
    kernel_init = 'glorot_uniform'
 
    wav_in = Input(shape=(window_length, featdim)) 
    enc_out = wav_in

    # Defining the Encoder
    for layernum, numkernels in enumerate(g_enc_numkernels):
        if layernum==0 and opts ['GT_init_G']:
            enc_out = Conv1D(num_gt_filters, gt_filterlength, 
                               kernel_initializer=kernel_init, strides=pool, 
                                   padding="same", use_bias=use_bias, 
                                        name="G_gtlayer")(enc_out)
        else :
            enc_out = Conv1D(numkernels, kwidth, strides=pool,
                               kernel_initializer=kernel_init, padding="same", 
                                   use_bias=use_bias)(enc_out)
      
        # for skip connections
        if layernum < len(g_enc_numkernels) - 1:
            skips.append(enc_out)
        if opts['applyprelu']:
            enc_out = PReLU(alpha_initializer='zero', weights=None)(enc_out)
        else:
            enc_out = LeakyReLU(alpha=opts['leakyrelualpha'])(enc_out)

    num_enc_layers = len(g_enc_numkernels)
    z_rows = int(window_length/ (pool ** num_enc_layers))
    z_cols = g_enc_numkernels[-1]

    # Adding the intermediate noise layer
    if not opts['z_off']:
        z = Input(shape=(z_rows,z_cols), name='noise_input')
        dec_out = keras.layers.concatenate([enc_out, z])
    else :
        dec_out = enc_out

    # Now to the decoder part
    nrows = z_rows
    ncols = dec_out.get_shape().as_list()[-1]
    for declayernum, decnumkernels in enumerate(g_dec_numkernels):
        # reshape for the conv2dtranspose layer as it needs 3D input
        indim = dec_out.get_shape().as_list()
        newshape = (indim[1], 1 , indim[2])
        dec_out = Reshape(newshape)(dec_out)
        # add the conv2dtranspose layer
        dec_out = Conv2DTranspose(decnumkernels, [kwidth,1], strides=[strides, 1],
                     kernel_initializer=kernel_init, padding="same", use_bias=use_bias)(dec_out)
        # Reshape back to 2D
        nrows *= strides # number of rows get multiplied by strides
        ncols = decnumkernels # number of cols is the same as number of kernels
        dec_out.set_shape([None, nrows, 1 , ncols]) # for correcting shape issue with conv2dtranspose
        newshape = (nrows, ncols)
        if declayernum == len(g_dec_numkernels) -1:
            dec_out = Reshape(newshape, name="g_output")(dec_out) # name the final output as  g_output
        else:
            dec_out = Reshape(newshape)(dec_out)

       # add skip and prelu until the second-last layer
        if declayernum < len(g_dec_numkernels) -1 :
            if opts['applyprelu']:
                dec_out = PReLU(alpha_initializer='zero', weights=None)(dec_out)
            else:
                dec_out = LeakyReLU(alpha=opts['leakyrelualpha'])(dec_out)
            # Now add the skip connection
            skip_ = skips[-(declayernum + 1)]
            dec_out = keras.layers.concatenate([dec_out, skip_])
      
    
    # Add tanh of G uses tanh activation
    if opts['Gtanh']:
        dec_out = Activation('tanh')(dec_out)     

    # Create the model graph
    if opts ['z_off']:
        G = Model(inputs=[wav_in], outputs=[dec_out])
    else :
        G = Model(inputs=[wav_in, z], outputs=[dec_out])

    # add GT initilization
    if opts['GT_init_G']:
        G.get_layer("G_gtlayer").set_weights([gt, gt_bias])
        # set it trainable or not
        if opts['gt_fixed']:
            G.get_layer("G_gtlayer").trainable = False
      
    if opts ['show_summary'] :
        G.summary()

    return G



def discriminator(opts):
    print('*** Building Discriminator ***')
    window_length = opts['window_length']
    featdim = opts ['featdim']
    batch_size = opts['batch_size']
    d_fmaps = opts ['d_fmaps']
    strides = opts['strides']
    activation = opts['d_activation']
    kwidth = opts['filterlength']
    # gammatone layer
    if opts['GT_init_D']:
        gt = np.expand_dims(opts['gt'], axis=1)
        num_gt_filters = gt.shape[2]
        gt_filterlength = gt.shape[0]
        gt_bias = np.zeros((num_gt_filters,))
        # we will need 2D filters since discriminator uses two inputs 
        gt_con = np.concatenate((gt, gt), axis=1)
        
    wav_in_clean = Input(shape=(window_length, featdim), name='disc_inputclean')
    wav_in_noisy = Input(shape=(window_length, featdim), name='disc_inputnoisy')

    use_bias= True
    #kernel_init = keras.initializers.TruncatedNormal(stddev=0.02)
    kernel_init = 'glorot_uniform'
    
    d_out = keras.layers.concatenate([wav_in_clean, wav_in_noisy])

    for layer_num, numkernels in enumerate(d_fmaps):
        if layer_num == 0 and opts['GT_init_D']:
            d_out = Conv1D(num_gt_filters, gt_filterlength, strides=strides, 
                        kernel_initializer=kernel_init, use_bias=use_bias, padding="same", name="gtlayer")(d_out)
        else :
            d_out = Conv1D(numkernels, kwidth, strides=strides, kernel_initializer=kernel_init, 
                        use_bias=use_bias, padding="same")(d_out)

        if opts['applybn']:
            d_out = BatchNormalization()(d_out)
        elif opts['applyinstancenorm'] :
            d_out = InstanceNormalization(axis=2)(d_out)

        if activation == 'leakyrelu':
            d_out = LeakyReLU(alpha=opts['leakyrelualpha'])(d_out)
        elif activation == 'relu':
            d_out = tf.nn.relu(d_out)

    d_out = Conv1D(1, 1, padding="same", use_bias=use_bias, kernel_initializer=kernel_init, 
                    name='logits_conv')(d_out)
    d_out = Flatten()(d_out)
    d_out = Dense(1, activation='linear', name='d_output')(d_out)
    D = Model([wav_in_clean, wav_in_noisy], d_out)
  
    # initialize with GT weights
    if opts['GT_init_D']:
        D.get_layer("gtlayer").set_weights([gt_con, gt_bias])
        if opts['gt_fixed']:
            D.get_layer("gtlayer").trainable = False

    if opts ['show_summary']:
        D.summary()
    return D

