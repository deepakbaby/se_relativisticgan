#!/bin/bash
# The dataset can be downloaded manually from
# https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791

# specify the location to which the database be copied
datadir='./data/' 


# adapted from https://github.com/santi-pdp/segan
datasets="clean_trainset_56spk_wav noisy_trainset_56spk_wav clean_testset_wav noisy_testset_wav"

# DOWNLOAD THE DATASET
mkdir -p $datadir
pushd $datadir


for dset in $datasets; do
    if [ ! -d ${dset}_16kHz ]; then
        # Clean utterances
        if [ ! -f ${dset}.zip ]; then
            echo 'DOWNLOADING $dset'
            wget http://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/${dset}.zip
        fi
        if [ ! -d ${dset} ]; then
            echo 'INFLATING ${dset}...'
            unzip -q ${det}.zip -d $dset
        fi
        if [ ! -d ${dset}_16kHz ]; then
            echo 'CONVERTING WAVS TO 16K...'
            mkdir -p ${dset}_16kHz
            pushd ${dset}
            ls *.wav | while read name; do
                sox $name -r 16k ../${dset}_16kHz/$name
            done
            popd
        fi
    fi
done

popd

# make a copy of the filelists in datadir
ls clean_trainset_56spk_wav/*.wav > $datadir/train_wav.txt
ls clean_testset_wav/*.wav > $datadir/test_wav.txt
