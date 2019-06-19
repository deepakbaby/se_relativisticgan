'''
Get filenames
'''
import tensorflow as tf


def get_modeldirname(opts):
    modeldir = opts['dirhead']

    # add if noise input is not there
    if opts ['z_off']:
        modeldir += "_noZ"
  
    # add normalization name
    if opts['applyinstancenorm']:
        modeldir += "_IN"
    elif opts['applybatchrenorm']:
        modeldir += "_BRN"
    elif opts['applybatchnorm']:
        modeldir += "_BN"
    elif opts['applygroupnorm']:
        modeldir += "_GN"
    elif opts['applyspectralnorm']:
        modeldir += "_SN"

    # add optimizer
    modeldir += "_Adam_D"
    modeldir += str(opts['d_lr'])
    modeldir += "_G"
    modeldir += str(opts['g_lr'])
  
    # add L1 norm
    modeldir += "_L1_" + str(opts ['g_l1loss'])
    return modeldir
    

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
  
