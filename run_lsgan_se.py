"""
Reimplementing SEGAN paper as close as possible in Keras. 
But use instance normalization instread of virtual batch normalization
Deepak Baby, UGent, June 2018.
"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, flatten, fully_connected
import numpy as np
import keras
from keras.layers import Input, Dense, Conv1D, Conv2D, Conv2DTranspose, BatchNormalization
from keras.layers import LeakyReLU, PReLU, Reshape, Concatenate, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
keras_backend = tf.keras.backend
keras_initializers = tf.keras.initializers
from data_ops import *
from file_ops import *
from models import *

import time
from tqdm import *
import h5py
import os,sys
import scipy.io.wavfile as wavfile

if __name__ == '__main__':

    # Various GAN options
    opts = {}
    opts ['dirhead'] = "LSGAN"
    opts ['z_off'] = True # set to True to omit the latent noise input
    # normalization
    #################################
    # Only one of the follwoing should be set to True
    opts ['applyinstancenorm'] = True
    opts ['applybn'] = False
    ##################################
    # Show model summary
    opts ['show_summary'] = True
   
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
    opts ['d_fmaps'] = opts ['g_enc_numkernels'] # We use the same structure for discriminator
    opts['leakyrelualpha'] = 0.3
    opts ['batch_size'] = 100
    opts ['applyprelu'] = True
    opts ['preemph'] = 0.95
   
    opts ['d_activation'] = 'leakyrelu'
    g_enc_numkernels = opts ['g_enc_numkernels']
    opts ['g_dec_numkernels'] = g_enc_numkernels[:-1][::-1] + [1]
    opts ['gt_stride'] = 2
    opts ['g_l1loss'] = 200.
    opts ['d_lr'] = 0.0002
    opts ['g_lr'] = 0.0002
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
        G = Model([wav_in_noisy, z], G_wav)
    else :
        G_wav = G(wav_in_noisy)
        G = Model(wav_in_noisy, G_wav)
 
    d_out = D([wav_in_clean, wav_in_noisy])
    D = Model([wav_in_clean, wav_in_noisy], d_out)
    G.summary()
    D.summary()

    # compile individual models
    D.compile(loss='mean_squared_error', optimizer=d_opt)
    G.compile(loss='mean_absolute_error', optimizer=g_opt)

    # for the combined model, we set the discriminator to be not trainable
    D.trainable = False
    D_out = D([G_wav, wav_in_noisy])
    if not opts ['z_off']:
        G_D = Model(inputs=[wav_in_clean, wav_in_noisy, z], outputs=[D_out, G_wav])
    else :
        G_D = Model(inputs=[wav_in_clean, wav_in_noisy], outputs=[D_out, G_wav])
    G_D.summary()

    G_D.compile(optimizer=g_opt,
              loss={'model_2': 'mean_absolute_error', 'model_4': 'mean_squared_error'},
              loss_weights = {'model_2' : opts['g_l1loss'], 'model_4': 1} )
    print (G_D.metrics_names)
    
    #exit ()
    
    if TEST_SEGAN:
        ftestnoisy = h5py.File(noisy_test_matfile)
        noisy_test_data = ftestnoisy['feat_data']
        noisy_test_dfi = ftestnoisy['dfi']
        print ("Number of test files: " +  str(noisy_test_dfi.shape[1]) )


    # Begin the training part
    if TRAIN_SEGAN:   
        fclean = h5py.File(clean_train_matfile)
        clean_train_data = np.array(fclean['feat_data']).astype('float32')
        fnoisy = h5py.File(noisy_train_matfile)
        noisy_train_data = np.array(fnoisy['feat_data']).astype('float32')
        numtrainsamples = clean_train_data.shape[1]
        idx_all = np.arange(numtrainsamples)
        # set random seed
        np.random.seed(opts['random_seed'])
        batch_size = opts['batch_size']

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
                    noiseinput = np.random.normal(0, 1, (batch_size, z_dim1, z_dim2))
                    g_out = G.predict([noisywavs, noiseinput])
                else :
                    g_out = G.predict(noisywavs)

                # train D
                d_loss_real = D.train_on_batch ({'main_input_clean':cleanwavs, 'main_input_noisy':noisywavs}, 
                                    opts['D_real_target'] * np.ones((batch_size,1)))
                d_loss_fake = D.train_on_batch ({'main_input_clean':g_out, 'main_input_noisy':noisywavs}, 
                                      np.zeros((batch_size,1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the combined model next; here, only the generator part is update
                valid_g = np.array([1]*batch_size) # generator wants discriminator to give 1 (identify fake as real)
                if not opts['z_off']:
                    [g_loss, g_dLoss, g_l1loss] = G_D.train_on_batch({'main_input_clean': cleanwavs,
                                                    'main_input_noisy': noisywavs, 'noise_input': noiseinput}, 
                                                          {'model_2': cleanwavs, 'model_4': valid_g} )
                else:
                    [g_loss, g_dLoss, g_l1loss] = G_D.train_on_batch({'main_input_clean': cleanwavs,
                                               'main_input_noisy': noisywavs},{'model_2': cleanwavs, 
                                                          'model_4': valid_g} )
                time_taken = time.time() - start_time

                printlog = "E%d/%d:B%d/%d [D loss: %f] [D real loss: %f] [D fake loss: %f] [G loss: %f] [G_D loss: %f] [G_L1 loss: %f] [Exec. time: %f]" %  (epoch, n_epochs, batch_idx, num_batches_per_epoch, d_loss, d_loss_real, d_loss_fake, g_loss, g_dLoss, g_l1loss, time_taken)
        
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
