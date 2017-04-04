#!/bin/sh
#PBS -S /bin/sh
#PBS -N build
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l mem=4gb
#PBS -q fluxg
#PBS -A eecs542w17_fluxg
#PBS -l qos=flux
#PBS -M erhsin@umich.edu
#PBS -m abe
#PBS -l walltime=1:00:00
#PBS -j eo
#PBS -V

module load matlab
module load ffmpeg
module load image-libraries
module load opencv
module load hdf5/1.8.16/gcc/4.8.5
module load cuda/7.5
module load cudnn/7.5-v5
module load mkl
module load caffe
