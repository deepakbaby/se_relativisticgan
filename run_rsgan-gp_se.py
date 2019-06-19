"""
Reimplementing segan paper as close as possible. 
Deepak Baby, UGent, June 2018.
"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, flatten, fully_connected
import numpy as np
from keras.layers import Subtract, Activation, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import _Merge
from keras.callbacks import TensorBoard
import keras.backend as K

from data_ops import *
from file_ops import *
from models import *
from wgan_ops import *
from functools import partial
import time
from tqdm import *
import h5py
import os,sys
import scipy.io.wavfile as wavfile

BATCH_SIZE = 100
GRADIENT_PENALTY_WEIGHT = 10 # need to tune

class RandomWeightedAverage (_Merge):
    def _merge_function (self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

if __name__ == '__main__':

    # Various GAN options
    opts = {}
    opts ['dirhead'] = 'RSGAN_GP' + str(GRADIENT_PENALTY_WEIGHT)
    opts ['gp_weight'] = GRADIENT_PENALTY_WEIGHT
    ##########################
    opts ['z_off'] = not False # set to True to omit the latent noise input
    # normalization
    #################################
    # Only one of the follwoing should be set to True or all of can be False
    opts ['applybn'] = False
    opts ['applyinstancenorm'] = True # Works even without any normalization
    ##################################
    # Show model summary
    opts ['show_summary'] = False
   
    ## Set the matfiles
    clean_train_matfile = "./data/clean_train_segan1d.mat"
    noisy_train_matfile = "./data/noisy_train_segan1d.mat"
    noisy_test_matfile = "./data/noisy_test_segan1d.mat"
 
    ####################################################
    # Other fixed options
    opts ['window_length'] =  2**14
    opts ['featdim'] = 1 # 1 since it is just 1d time samples
    opts ['filterlength'] =  31
    opts ['strides'] = 2
    opts ['padding'] = 'SAME'
    opts ['g_enc_numkernels'] = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    opts ['g_enc_lstm_cells'] = [1024]
    opts ['d_fmaps'] = opts ['g_enc_numkernels'] # We use the same structure for discriminator
    opts ['d_lstms'] = opts ['g_enc_lstm_cells']
    opts['leakyrelualpha'] = 0.3
    opts ['batch_size'] = BATCH_SIZE
    opts ['applyprelu'] = True

   
    opts ['d_activation'] = 'leakyrelu'
    g_enc_numkernels = opts ['g_enc_numkernels']
    opts ['g_dec_numkernels'] = g_enc_numkernels[:-1][::-1] + [1]
    opts ['gt_stride'] = 2
    opts ['g_l1loss'] = 200.
    opts ['d_lr'] = 2e-4
    opts ['g_lr'] = 2e-4
    opts ['random_seed'] = 111
 
    n_epochs = 81
    fs = 16000
    
    # set flags for training or testing
    TRAIN_SEGAN =  True
    SAVE_MODEL =  True
    LOAD_SAVED_MODEL = False
    TEST_SEGAN =  True

    modeldir = get_modeldirname(opts)
    print ("The model directory is " + modeldir)
    print ("_____________________________________")

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # Obtain the generator and the discriminator
    D = discriminator(opts)
    G = generator(opts)

    # Define optimizers
    g_opt = keras.optimizers.Adam(lr=opts['g_lr'])
    d_opt = keras.optimizers.Adam(lr=opts['d_lr'])

    # The G model has the wav and the noise inputs
    wav_shape = (opts['window_length'], opts['featdim'])
    z_dim1 = int(opts['window_length']/ (opts ['strides'] ** len(opts ['g_enc_numkernels'])))
    z_dim2 = opts ['g_enc_numkernels'][-1]
    wav_in_clean =  Input(shape=wav_shape, name="main_input_clean")
    wav_in_noisy = Input(shape=wav_shape, name="main_input_noisy")
    if not opts ['z_off']:
        z = Input (shape=(z_dim1, z_dim2), name="noise_input")
        G_wav = G([wav_in_noisy, z])
        G_model = Model([wav_in_noisy, z], G_wav)
    else :
        G_wav = G(wav_in_noisy)
        G_model = Model(wav_in_noisy, G_wav)
 
    d_out = D([wav_in_clean, wav_in_noisy])
    D = Model([wav_in_clean, wav_in_noisy], d_out)
    G_model.summary()
    D.summary()

    # ADDING RELATIVISTIC LOSS AT OUTPUT
    for layer in D.layers :
        layer.trainable = False
    D.trainable = False
    if not opts ['z_off']:
        G_wav = G([wav_in_noisy, z])
    else :
        G_wav = G(wav_in_noisy)
    D_out_for_G = D([G_wav, wav_in_noisy])
    D_out_for_real = D([wav_in_clean, wav_in_noisy])

    d_outG = Subtract()([D_out_for_G, D_out_for_real])
    d_outG = Activation('sigmoid', name="DoutG")(d_outG)

    if not opts ['z_off']:
        G_D =  Model(inputs=[wav_in_clean, wav_in_noisy, z], outputs = [d_outG, G_wav])
    else :
        G_D =  Model(inputs=[wav_in_clean, wav_in_noisy], outputs = [d_outG, G_wav])
    
    G_D.summary()
    G_D.compile(optimizer=g_opt,
              loss={'model_2': 'mean_absolute_error', 'DoutG': 'binary_crossentropy'},
              loss_weights = {'model_2' : opts['g_l1loss'], 'DoutG': 1} )
    print (G_D.metrics_names)

    # Now we need D model so that gradient penalty can be incorporated
    for layer in D.layers :
        layer.trainable = True
    for layer in G.layers :
        layer.trainable = False
    D.trainable = True
    G.trainable = False
    if not opts ['z_off']:
        G_wav_for_D =  G([wav_in_noisy, z])
    else :
        G_wav_for_D = G(wav_in_noisy)
   
    d_out_for_G = D([G_wav_for_D, wav_in_noisy])
    d_out_for_real = D([wav_in_clean, wav_in_noisy])
    # for gradient penalty
    averaged_samples = RandomWeightedAverage()([wav_in_clean, G_wav_for_D])
    # We will need to this also through D, for computing the gradients
    d_out_for_averaged = D([averaged_samples, wav_in_noisy])
    # compute the GP loss by means of partial function in keras
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples = averaged_samples,
                               gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'
    
    d_outD = Subtract()([d_out_for_real, d_out_for_G])
    d_outD = Activation('sigmoid', name="DoutD")(d_outD)
    
    if not opts ['z_off']:
        D_final = Model(inputs = [wav_in_clean, wav_in_noisy, z], 
                        outputs = [d_outD, d_out_for_averaged])
    else :
        D_final = Model(inputs = [wav_in_clean, wav_in_noisy],
                        outputs = [d_outD, d_out_for_averaged])
    D_final.compile(optimizer = d_opt, 
                    loss = ['binary_crossentropy', partial_gp_loss ])

    D_final.summary()
    print (D_final.metrics_names)
    
    # create label vectors for training
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -1 * positive_y
    dummy_y =  np.zeros((BATCH_SIZE, 1), dtype=np.float32) # for GP Loss

    if TEST_SEGAN:
        ftestnoisy = h5py.File(noisy_test_matfile)
        noisy_test_data = ftestnoisy['feat_data']
        noisy_test_dfi = ftestnoisy['dfi']
        print ("Number of test files: " +  str(noisy_test_dfi.shape[1]) )


    # Begin the training part
    if TRAIN_SEGAN:
        fclean = h5py.File(clean_train_matfile)
        clean_train_data = np.array(fclean['feat_data'])
        fnoisy = h5py.File(noisy_train_matfile)
        noisy_train_data = np.array(fnoisy['feat_data'])
        print ("********************************************")
        print ("               SEGAN TRAINING               ")
        print ("********************************************")
        print ("Shape of clean feats mat " + str(clean_train_data.shape))
        print ("Shape of noisy feats mat " + str(noisy_train_data.shape))
        numtrainsamples = clean_train_data.shape[1]

        # Tensorboard stuff
        log_path = './logs/' + modeldir
        callback = TensorBoard(log_path)
        callback.set_model(G_D)
        train_names = ['G_loss', 'G_adv_loss', 'G_l1Loss']

        idx_all = np.arange(numtrainsamples)
        # set random seed
        np.random.seed(opts['random_seed'])

        batch_size = opts['batch_size']
        num_batches_per_epoch = int(np.floor(clean_train_data.shape[1]/batch_size))
        for epoch in range(n_epochs):
            # train D with  minibatch
            np.random.shuffle(idx_all) # shuffle the indices for the next epoch
            for batch_idx in range(num_batches_per_epoch):
                start_time = time.time()
                idx_beg = batch_idx * batch_size
                idx_end = idx_beg + batch_size
                idx = np.sort(np.array(idx_all[idx_beg:idx_end]))
                #print ("Batch idx " + str(idx[:5]) +" ... " + str(idx[-5:]))
                cleanwavs = np.array(clean_train_data[:,idx]).T
                cleanwavs = data_preprocess(cleanwavs, preemph=opts['preemph'])
                cleanwavs = np.expand_dims(cleanwavs, axis = 2)
                noisywavs = np.array(noisy_train_data[:,idx]).T
                noisywavs = data_preprocess(noisywavs, preemph=opts['preemph'])
                noisywavs = np.expand_dims(noisywavs, axis = 2)
                if not opts ['z_off']:
                    noiseinput = np.random.normal(0, 1, 
                                                 (batch_size, z_dim1, z_dim2))
                    [_, d_loss, d_gploss] = D_final.train_on_batch({'main_input_clean': cleanwavs,
                                                    'main_input_noisy': noisywavs, 'noise_input': noiseinput},
                                                     {'DoutD': positive_y, 'model_4': dummy_y} ) 
                    [g_loss, g_dLoss, g_l1loss] = G_D.train_on_batch({'main_input_clean': cleanwavs,
                                                    'main_input_noisy': noisywavs, 'noise_input': noiseinput}, 
                                                          {'model_2': cleanwavs, 'DoutG': positive_y} )
                else:
                    [_, d_loss, d_gploss] = D_final.train_on_batch({'main_input_clean': cleanwavs,
                                                      'main_input_noisy': noisywavs,},
                                                      {'DoutD': positive_y, 'model_4': dummy_y} )
                    [g_loss, g_dLoss, g_l1loss] = G_D.train_on_batch({'main_input_clean': cleanwavs,
                                                                      'main_input_noisy': noisywavs},
                                                                      {'model_2': cleanwavs, 
                                                                        'DoutG': positive_y} )
                time_taken = time.time() - start_time

                printlog = "E%d/%d:B%d/%d [D loss: %f] [D_GP loss: %f] [G loss: %f] [G_D loss: %f] [G_L1 loss: %f] [Exec. time: %f]" %  (epoch, n_epochs, batch_idx, num_batches_per_epoch, d_loss, d_gploss, g_loss, g_dLoss, g_l1loss, time_taken)
        
                print (printlog)
                # Tensorboard stuff 
                logs = [g_loss, g_dLoss, g_l1loss]
                write_log(callback, train_names, logs, epoch)

            if (TEST_SEGAN and epoch % 10 == 0) or epoch == n_epochs - 1:
                print ("********************************************")
                print ("               SEGAN TESTING                ")
                print ("********************************************")

                resultsdir = modeldir + "/test_results_epoch" + str(epoch)
                if not os.path.exists(resultsdir):
                    os.makedirs(resultsdir)

                if LOAD_SAVED_MODEL:
                    print ("Loading model from " + modeldir + "/Gmodel")
                    json_file = open(modeldir + "/Gmodel.json", "r")
                    loaded_model_json = json_file.read()
                    json_file.close()
                    G_loaded = model_from_json(loaded_model_json)
                    G_loaded.compile(loss='mean_squared_error', optimizer=g_opt)
                    G_loaded.load_weights(modeldir + "/Gmodel.h5")
                else:
                    G_loaded = G

                print ("Saving Results to " + resultsdir)

                for test_num in tqdm(range(noisy_test_dfi.shape[1])) :
                    test_beg = noisy_test_dfi[0, test_num]
                    test_end = noisy_test_dfi[1, test_num]
                    #print ("Reading indices " + str(test_beg) + " to " + str(test_end))
                    noisywavs = np.array(noisy_test_data[:,test_beg:test_end]).T
                    noisywavs = data_preprocess(noisywavs, preemph=opts['preemph'])
                    noisywavs = np.expand_dims(noisywavs, axis = 2)
                    if not opts['z_off']:
                        noiseinput = np.random.normal(0, 1, (noisywavs.shape[0], z_dim1, z_dim2))
                        cleaned_wavs = G_loaded.predict([noisywavs, noiseinput])
                    else :
                        cleaned_wavs = G_loaded.predict(noisywavs)
          
                    cleaned_wavs = np.reshape(cleaned_wavs, (noisywavs.shape[0], noisywavs.shape[1]))
                    cleanwav = reconstruct_wav(cleaned_wavs)
                    cleanwav = np.reshape(cleanwav, (-1,)) # make it to 1d by dropping the extra dimension
                    
                    if opts['preemph'] > 0:
                        cleanwav = de_emph(cleanwav, coeff=opts['preemph'])

                    destfilename = resultsdir +  "/testwav_%d.wav" % (test_num)
                    wavfile.write(destfilename, fs, cleanwav)



        # Finally, save the model
        if SAVE_MODEL:
            model_json = G.to_json()
            with open(modeldir + "/Gmodel.json", "w") as json_file:
                json_file.write(model_json)
            G.save_weights(modeldir + "/Gmodel.h5")
            print ("Model saved to " + modeldir)
