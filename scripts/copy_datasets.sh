#!/bin/bash

copy_imagenet(){
    # Args:
    # $1: (str) SRC_DIR path

    source $1/.env
    COPY_START=$(date +%s.%N)

    echo "Copying Imagenet data to SLURM_TMPDIR: {$SLURM_TMPDIR}"
    mkdir -p $SLURM_TMPDIR/imagenet
    cp -r -n /network/datasets/imagenet.var/imagenet_torchvision/val $SLURM_TMPDIR/imagenet
    mkdir -p $SLURM_TMPDIR/imagenet/train
    cd       $SLURM_TMPDIR/imagenet/train
    tar  -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'

    mkdir -p $SLURM_TMPDIR/imagenet/scripts
    cd $SLURM_TMPDIR/imagenet/scripts
    cp -r -n /network/datasets/imagenet.var/imagenet_torchvision/scripts $SLURM_TMPDIR/imagenet/scripts
    cp -n /network/datasets/imagenet.var/imagenet_torchvision/meta.bin $SLURM_TMPDIR/imagenet/meta.bin

    COPY_END=$(date +%s.%N)
    DIFF=$(echo "$COPY_END - $COPY_START" | bc)
    echo "Total elapsed copy and extract time:" $DIFF "seconds"
    cd $1
}

copy_imagenet_debug(){
    # Args:
    # $1: (str) SRC_DIR path

    source $1/.env
    COPY_START=$(date +%s.%N)

    echo "Copying Imagenet debug data to SLURM_TMPDIR: {$SLURM_TMPDIR}"
#    for split in train val; do
#        for sub_dir in /network/datasets/imagenet.var/imagenet_torchvision/"$split"/*/; do
#            for file in "$sub_dir"/*; do
#                mkdir -p "$SLURM_TMPDIR/imagenet/$split/$(basename "$sub_dir")" && cp -n "$file" "$_"
#                break
#            done
#        done
#    done

#    cp -n /network/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz "$SLURM_TMPDIR"/imagenet

    # The debug dataset was created by the above commented code
    rsync -a "$DATA_DIR"/imagenet_debug.tar.gz "$SLURM_TMPDIR"/
    tar --skip-old-files -xzf "$SLURM_TMPDIR"/imagenet_debug.tar.gz -C "$SLURM_TMPDIR"

    COPY_END=$(date +%s.%N)
    DIFF=$(echo "$COPY_END - $COPY_START" | bc)
    echo "Total elapsed copy and extract time:" $DIFF "seconds"
}

copy_mnist(){
    # Args:
    # $1: (str) SRC_DIR path

    source $1/.env

    COPY_START=$(date +%s.%N)

    echo "Copying MNIST data to SLURM_TMPDIR: {$SLURM_TMPDIR}"
    rsync -a "$DATA_DIR"/mnist "$SLURM_TMPDIR"/

    COPY_END=$(date +%s.%N)
    DIFF=$(echo "$COPY_END - $COPY_START" | bc)
    echo "Total elapsed copy and extract time:" $DIFF "seconds"
}

copy_cifar100(){
    # Args:
    # $1: (str) SRC_DIR path

    source $1/.env

    COPY_START=$(date +%s.%N)

    echo "Copying CIFAR100 data to SLURM_TMPDIR: {$SLURM_TMPDIR}"
    rsync -a /network/datasets/cifar100.var/cifar100_torchvision "$SLURM_TMPDIR"/
    mv "$SLURM_TMPDIR"/cifar100_torchvision "$SLURM_TMPDIR"/cifar100

    COPY_END=$(date +%s.%N)
    DIFF=$(echo "$COPY_END - $COPY_START" | bc)
    echo "Total elapsed copy and extract time:" $DIFF "seconds"
}

copy_dataset(){
    # Args:
    # $1: (str) $DATASET_NAME
    # $2: (str) SRC_DIR path
    # $3: (int) $DEBUG_DATASET (only for imagenet). Default: 0

    if [[ $1 == "mnist" ]]; then
        copy_mnist $2
    elif [[ $1 == "imagenet" ]]; then
        if (( $3 )); then
            copy_imagenet_debug $2
        else
            copy_imagenet $2
        fi
    elif [[ $1 == "cifar100" ]]; then
        copy_cifar100 $2
    else
        echo "Unknown dataset: $1"
        exit 1
    fi
}
