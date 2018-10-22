'''
Get filenames
'''
import tensorflow as tf


def get_modeldirname(opts):
    modeldir = opts['dirhead']

    # add if G uses tanh
    if opts ['Gtanh']:
        modeldir += "_Gtanh"

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

    # add labelsmooth
    if opts['D_real_target'] < 1.0 :
        modeldir += "_LabSmth" + str(opts['D_real_target'])

    # add gammatone init
    if opts['GT_init_G'] and opts['GT_init_D'] :
        modeldir += "_GDgt"
    elif opts['GT_init_D'] :
        modeldir += "_Dgt"
    elif opts['GT_init_G'] :
        modeldir += "_Ggt"
    
    if opts['GT_init_G'] or opts['GT_init_D'] :
        if opts['gt_fixed']:
            modeldir += "fixed"

        gt = opts['gt']
        num_gt_filters = gt.shape[1]
        gt_filterlength = gt.shape[0]
        addstring = str(num_gt_filters) + "L" + str(gt_filterlength)
        addstring += "stride" 
        addstring += str(opts['gt_stride'])
        modeldir += addstring

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
  
