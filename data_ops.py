"""
Data processing routines
Deepak Baby, UGent, June 2018
deepak.baby@ugent.be
"""

import numpy as np

def reconstruct_wav(wavmat, stride_factor=0.5):
  """
  Reconstructs the audiofile from sliced matrix wavmat
  """
  window_length = wavmat.shape[1]
  window_stride = int(stride_factor * window_length)
  wav_length = (wavmat.shape[0] -1 ) * window_stride + window_length
  wav_recon = np.zeros((1,wav_length))
  #print ("wav recon shape " + str(wav_recon.shape))
  for k in range (wavmat.shape[0]):
    wav_beg = k * window_stride
    wav_end = wav_beg + window_length
    wav_recon[0, wav_beg:wav_end] += wavmat[k, :]

  # now compute the scaling factor for multiple instances
  noverlap = int(np.ceil(1/stride_factor))
  scale_ = (1/float(noverlap)) * np.ones((1, wav_length))
  for s in range(noverlap-1):
    s_beg = s * window_stride
    s_end = s_beg + window_stride
    scale_[0, s_beg:s_end] = 1/ (s+1)
    scale_[0, -s_beg - 1 : -s_end:-1] = 1/ (s+1)

  return wav_recon * scale_

def pre_emph(x, coeff=0.95):
  """
  Apply pre_emph on 2d data (batch_size x window_length)
  """
  #print ("x shape: " +  str(x.shape))
  x0 = x[:, 0]
  x0 = np.expand_dims(x0, axis=1)
  diff = x[:, 1:] - coeff * x[:, :-1]
  x_preemph = np.concatenate((x0, diff), axis=1)
  if not x.shape == x_preemph.shape:
    print ("ERROR: Pre-emphasis is wrong")
  #print ("x_preemph shape: " +  str(x_preemph.shape))
  return x_preemph

def de_emph(y, coeff=0.95):
  """
    Apply de_emphasis on test data: works only on 1d data
  """
  if coeff <= 0:
    return y
  x = np.zeros((y.shape[0],), dtype=np.float32)
  #print("in_shape" + str(y.shape))
  x[0] = y[0]
  for n in range(1, y.shape[0], 1):
    x[n] = coeff * x[n - 1] + y[n]
  return x

def data_preprocess(wav, preemph=0.95):
  wav = (2./65535.) * (wav.astype('float32') - 32767) + 1.
  if preemph > 0:
    wav = pre_emph(wav, coeff=preemph)
  return wav.astype('float32')
