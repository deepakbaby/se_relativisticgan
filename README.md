# Keras framework for speech enhancement using relativistic GANs.
Uses a fully convolutional end-to-end speech enhancement system.

Implemetation details of the paper accepted to ICASSP-2019

**Deepak Baby and Sarah Verhulst, _SERGAN: Speech enhancement using relativistic generative adversarial networks with gradient penalty_, IEEE-ICASSP, pp. 106-110, May 2019, Brighton, UK.**

> This work was funded with support from the EU Horizon 2020 programme under grant agreement No 678120 (RobSpear).

__!!! Under Construction !!!__

----
### Pre-requisites
1. Install [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/)
2. The experiments are conducted on a dataset from Valentini et. al.,  and are downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/1942). The following script can be used to download the dataset. *Requires [sox](http://sox.sourceforge.net/) for converting to 16kHz*.
    ```bash
    $ ./download_dataset.sh
    ```

### Running the model
1. **Prepare data for training and testing the various models**. The folder path may be edited if you keep the database in a different folder. This script is to be executed only once and the all the models reads from the same location.
    ```python
    python prepare_data.py
    ```
2. **Running the models**. The models available in this repository are listed below. Every implementation offers several cGAN configurations. Edit the ```opts``` variable for choosing the cofiguration. The results will be automatically saved to different folders. The folder name is generated from ```files_ops.py ``` and the foldername automatically includes different configuration options.
    1. `run_aecnn.py`        : Auto-encoder CNN model with L1 loss term (No discriminator)
    1. `run_lsgan_se.py`     : SEGAN with least-squares loss [1]
    2. `run_wgan-gp_se.py`   : GAN model with Wassterstein loss and Gradient Penalty
    3. `run_rsgan-gp_se.py`  : GAN model with relativistic standard GAN with Gradient Penalty
    4. `run_rasgan-gp_se.py` : GAN model with relativistic average standard GAN with Gradient Penalty
    5. `run_ralsgan-gp_se.py`: GAN model with relativistic average least-squares GAN with Gradient Penalty

3. **Evaluation on testset is also done together with training**. Set ```TEST_SEGAN = False``` for disabling testing. 

----
### Misc
* **This code loads all the data into memory for speeding up training**. But if you dont have enough memory, it is possible  to read the mini-batches from the disk using HDF5 read. In ```run_<xxx>.py``` 
  ```python
  clean_train_data = np.array(fclean['feat_data'])
  noisy_train_data = np.array(fnoisy['feat_data'])
  ```
  change the above lines to 
  ```python
  clean_train_data = fclean['feat_data']
  noisy_train_data = fnoisy['feat_data']
  ```
  **But this can lead to a slow-down of about 20 times (on the test machine)** as the mini-batches are to be read from the disk over several epochs.

---- 
### References
[1] S. Pascual, A. Bonafonte, and J. Serra, _SEGAN: speech enhancement generative adversarial network_, in INTERSPEECH., ISCA, Aug 2017, pp. 3642–3646.

----
#### Credits
The keras implementation of cGAN is based on the following repos
* [SEGAN](https://github.com/santi-pdp/segan)
* [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
* [pix2pix](https://github.com/phillipi/pix2pix)

