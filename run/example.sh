# config
while getopts d:s:g: flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        d) data_dir=${OPTARG};;
    esac
done

# generate noise
for split in train test
do
    python 00_generate_noise.py \
        --data-dir ${data_dir}/datasets/cifar10 \
        --dataset cifar10 \
        --split ${split} \
        --noise-type uniform \
        --noise-rate 0.2 \
        --seed 19 \
        --save-path ${data_dir}/noise/cifar10_${split}_uniform_20_noise.npy
done

# run training
python 01_train_noise_detection.py \
    --data ${data_dir}/datasets/cifar10 \
    --noise ${data_dir}/noise/cifar10_train_uniform_20_noise.npy \
    --test-noise ${data_dir}/noise/cifar10_test_uniform_20_noise.npy \
    --seed 19 \
    --gpu ${gpu} \
    --save-dir ${data_dir}/models/standard/cifar10_resnet18_uniform_20_noise

# get baseline predictions
python 02a_create_filter_logit.py \
    --data ${data_dir}/datasets/cifar10 \
    --noise ${data_dir}/noise/cifar10_train_uniform_20_noise.npy \
    --model-dir ${data_dir}/models/standard/cifar10_resnet18_uniform_20_noise \
    --save-dir ${data_dir}/models/standard/cifar10_resnet18_uniform_20_noise

# get bgss predictions
python 02b_create_filter_bgss.py \
    --data ${data_dir}/datasets/cifar10 \
    --noise ${data_dir}/noise/cifar10_train_uniform_20_noise.npy \
    --model-dir ${data_dir}/models/standard/cifar10_resnet18_uniform_20_noise \
    --save-dir ${data_dir}/models/standard/cifar10_resnet18_uniform_20_noise

# run plateau training for plateau predictions
python 03_train_noise_detection_plateau.py \
    --data ${data_dir}/datasets/cifar10 \
    --noise ${data_dir}/noise/cifar10_train_uniform_20_noise.npy \
    --test-noise ${data_dir}/noise/cifar10_test_uniform_20_noise.npy \
    --seed 19 \
    --gpu ${gpu} \
    --save-dir ${data_dir}/models/plateau/cifar10_resnet18_uniform_20_noise

# train model on filtered dataset
python 04_train_clean_model.py \
    --data ${data_dir}/datasets/cifar10 \
    --noise ${data_dir}/noise/cifar10_train_uniform_20_noise.npy \
    --test-noise ${data_dir}/noise/cifar10_test_uniform_20_noise.npy \
    --train-idxs ${data_dir}/models/standard/cifar10_resnet18_uniform_20_noise/filter_baseline_top10_vote.npy \
    --seed 19 \
    --gpu ${gpu} \
    --save-dir ${data_dir}/models/cleaned/cifar10_resnet18_uniform_20_noise
