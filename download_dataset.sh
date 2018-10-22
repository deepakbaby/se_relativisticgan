#!/bin/bash


# specify the location to which the database be copied
datadir = './data/' 


# adapted from https://github.com/santi-pdp/segan
datasets="clean_trainset_wav noisy_trainset_wav clean_testset_wav noisy_testset_wav"

# DOWNLOAD THE DATASET
mkdir -p $datadir
pushd $datadir

for dset in datasets; do
    if [ ! -d ${dset}_16k ]; then
        # Clean utterances
        if [ ! -f ${dset}.zip ]; then
            echo 'DOWNLOADING $dset'
            wget http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/${dset}.zip
        fi
        if [ ! -d ${dset} ]; then
            echo 'INFLATING ${dset}...'
            unzip -q ${det}.zip -d $dset
        fi
        if [ ! -d ${dset}_16k ]; then
            echo 'CONVERTING WAVS TO 16K...'
            mkdir -p ${dset}_16k
            pushd ${dset}
            ls *.wav | while read name; do
                sox $name -r 16k ../${dset}_16k/$name
            done
            popd
        fi
    fi
done

popd

# make a copy of the filelists in datadir
cp train_wav.txt test_wav.txt $datadir
