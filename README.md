# Keras framework for speech enhancement using relativistic GANs.
Uses a fully convolutional end-to-end speech enhancement system.

Implemetation details of the paper submitted to ICASSP-2019

**Deepak Baby and Sarah Verhulst, _SERGAN: Speech enhancement using relativistic generative adversarial networks with gradient penalty_, Submitted to IEEE-ICASSP 2019.**

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
2. **Running the models**. The models available in this repository are listed below.
    1. `run_aecnn.py`        : Auto-encoder CNN model with L1 loss term (No discriminator)
    1. `run_lsgan_se.py`     : SEGAN with least-squares loss [1]
    2. `run_wgan-gp_se.py`   : GAN model with Wassterstein loss and Gradient Penalty
    3. `run_rsgan-gp_se.py`  : GAN model with relativistic standard GAN with Gradient Penalty
    4. `run_rasgan-gp_se.py` : GAN model with relativistic average standard GAN with Gradient Penalty
    5. `run_ralsgan-gp_se.py`: GAN model with relativistic average least-squares GAN with Gradient Penalty
----
### References
[1] S. Pascual, A. Bonafonte, and J. Serra, _SEGAN: speech enhancement generative adversarial network_, in INTERSPEECH., ISCA, Aug 2017, pp. 3642–3646.

----
#### Credits
The keras implementation of cGAN is based on the following repos
* [SEGAN](https://github.com/santi-pdp/segan)
* [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
* [pix2pix](https://github.com/phillipi/pix2pix)

