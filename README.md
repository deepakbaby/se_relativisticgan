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
    
#### Credits
The keras implementation of cGAN is based on the following repos
* [SEGAN](https://github.com/santi-pdp/segan)
* [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
* [pix2pix](https://github.com/phillipi/pix2pix)

