# config
while getopts d:s:g: flag
do
    case "${flag}" in
        d) data_dir=${OPTARG};;
        g) gpu=${OPTARG};;
        s) seed=${OPTARG};;
    esac
done

# training
python train.py \
    --data $data_dir/datasets/cifar10 \
    --gpu $gpu \
    --seed $seed \
    --save-dir $data_dir/models/train/cifar10_resnet18
