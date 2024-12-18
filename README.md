# Deep Model Reassembly  🏭 -> 🧱 -> 🏭
## 😎 Introduction
This repository contains the offical implementation for our paper

**Deep Model Reassembly** (NeurIPS2022)

[[arxiv](https://arxiv.org/abs/2210.17409)] [[project page](https://adamdad.github.io/dery/)]
 [[code](https://github.com/Adamdad/DeRy)]

*Xingyi Yang, Daquan Zhou, Songhua Liu, Jingwen Ye, Xinchao Wang*

> In this work, we explore a novel knowledge-transfer task, termed as Deep Model Reassembly (*DeRy*), for general-purpose model reuse. *DeRy* first dissect each model into distinctive building blocks, and then selectively reassemble the derived blocks to produce customized networks under both the hardware resource and performance constraints.

![pipeline](assets/pipeline.png)

- [x] 2022/12/07 Code Updated for better usage.

## 📚 File Organization

```
DeRy/
├── blocklize/block_meta.py         [Meta Information & Node Defnition]

├── similarity/
│   ├── get_rep.py                  [Compute and save the feature embeddings]
│   ├── get_sim.py                  [Compute representation similarity given the saved features]
|   ├── partition.py                [Network partition by cover set problem]
|   ├── zeroshot_reassembly.py      [Network reassembly by solving integer program]

├── configs/
|   ├── compute_sim/                [Model configs in the model zoo to compute the feature similarity]
|   ├── dery/XXX/$ModelSize_$DataSet_$BatchSize_$TrainTime_dery_$Optimizor.py   [Config files for transfer experiments]

├── mmcls_addon/
|   ├── datasets/                   [Dataset definitions]
|   ├── models/backbones/dery.py    [DeRy backbone definition]

├── third_package/timm              [Modified timm package]
```
    

## 🛠 Installation
The model training part is based on [mmclassification](https://github.com/open-mmlab/mmclassification). Some of the pre-trained weights are from [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm).

    # Create python env
    conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
    conda activate open-mmlab

    # Install mmcv and mmcls
    # CAUTION: You may need to use `pip install` instead of `mim install`. `mim` can sometimes enter an infinite loop.
    pip3 install openmim
    mim install mmcv==1.4.8
    mim install mmcls==0.21.0

    # Install timm
    pip install timm==0.6.13
    
    # Install older versions of these packages
    pip install matplotlib==3.6.3
    pip install numpy==1.26.4
    pip install yapf==0.32.0


**Note:** Our code needs `torch.fx` to support the computational graph extraction from the torch model. Therefore, please install the `torch > 1.10`.


## 🚀 Getting Started
To run the code for *DeRy*, we need to go through 4 steps. To test only the model stitching/fine-tuning, skip to Step 4.
The *DeRy* outputs from the paper (the results of Stage 3, before fine-tuning) are provided at [configs/dery](configs/dery).

**Note**: Partitioning and reassembling results may not be identical because of the algorithmatic stochasticity. It may
slightly affect the performance.


1. [**Model Zoo Preparation**] Compute the model feature embeddings and representation similarity. We first write model configuration and its weight path, and run the configs in `configs/compute_sim`
            
        PYTHONPATH="$PWD" python simlarity/get_rep.py \
        $Config_file \              # configs in `configs/compute_sim`
        --root $Feature_dir \       # Save feature embeddings in *.pth files
        [--checkpoint $Checkpoint]  # download checkpoint if any

    For example:

        PYTHONPATH="$PWD" python simlarity/get_rep.py configs/compute_sim/resnet18_imagenet.py --root outputs/features

    All feature embeddings need to be saved in *.pth* files in the same `$Feature_dir` folder. We then load them and
    compute the feature similarity. Similarity will be saved as *net1.net2.pkl* files.

        PYTHONPATH="$PWD" python simlarity/compute_sim.py \
        --feat_path $Feature_dir \
        --sim_func $Similarity_function [cka, rbf_cka, lr]

    For example:

        PYTHONPATH="$PWD" python simlarity/compute_sim.py --feat_path outputs/features --sim_func cka --out outputs/sim

    Pre-computed similarities on ImageNet are available for [Linear CKA](https://drive.google.com/drive/folders/1ebSVwZyKeHdmdOdVlFZF6P9_1PzEMs-J?usp=share_link) and [Linear Regression](https://drive.google.com/drive/folders/1rKmV3iQwETKBO3yYlsXFIxrPfAc7VRRb?usp=share_link).

    We also need to compute the feature size (input-output feature dimensions). It can be done by running

        PYTHONPATH="$PWD" python simlarity/count_inout_size.py --root $Feature_dir

    The results is a json file containing the input-output shape for all network layers, like [MODEL_INOUT_SHAPE.json](https://drive.google.com/file/d/15xDgYOu8Gs866faNHEYI0iHCJkxl-M2h/view?usp=share_link).
    However, a suitable file can already be found at [here](tools/MODEL_INOUT_SHAPE.json).

2. [**Network Partition**] Solve the cover set optimization to get the network partition. The results is an assignment file in *.pkl*.

        PYTHONPATH="$PWD" python simlarity/partition.py \
        --sim_path $Feat_similarity_path \
        --out      $Partition_out_dir \
        --K        $Num_partition \             # default=4
        --trial    $Num_repeat_runs \           # default=200
        --eps      $Size_ratio_each_block \     # default=0.2
        --num_iter $Maximum_num_iter_eachrun    # default=200

    For example, to use the default settings as done in the paper:

        mkdir outputs/partition
        PYTHONPATH="$PWD" python simlarity/partition.py --sim_path outputs/sim --out outputs/partition

3. [**Reassemby**] Reassemble the partitioned building blocks into a full model, by solving a integer program with training-free proxy. The results are a series of model configs in *.py*.

        PYTHONPATH="$PWD" python simlarity/zeroshot_reassembly.py \
        --path          $Block_partition_file [Saved in the partition step] \
        --C             $Maximum_parameter_num \
        --minC          $Minimum_parameter_num \
        --flop_C        $Maximum_FLOPs_num \
        --minflop_C     $Minimum_FLOPs_num \
        --num_batch     $Number_batch_average_to_compute_score \
        --batch_size    $Number_sample_each_batch \
        --trial         $Search_time \
        --zero_proxy    $Type_train_free_proxy [Default NASWOT] \
        --data_config   $Config_target_data

    For example, to use the default settings as done in the paper:

        PYTHONPATH="$PWD" python simlarity/zeroshot_reassembly.py --path outputs/partition/assignment_hybrid_4.pkl --zero_proxy naswot

4. [**Fune-Tuning**] Train the reassembled model on target data. The models from the paper are provided at
   [configs/dery](configs/dery). To train your own model, assemble a config similar to these examples, then run them
   using the training script. For example, the following would train the `DeRy(4,10,3)-FZ` model from Table 7.
   
        PYTHONPATH="$PWD" python tools/train.py configs/dery/imagenet/10m_imagenet_128x8_100e_dery_adamw_freeze.py
   
   The configs needed to reproduce Figure 11 (transfer learning) can be found in the `configs/dery/*_downstream` folders.
   **IMPORTANT:** Be sure to edit the config's `workers_per_gpu` parameter inside the `data` variable according to your
   machine specs. It would be best to assign 8-10 workers per GPU, if your machine has this many cores per GPU.
   
   The above shows the basic training script, but that can only use a single GPU, which will take many days to train.
   Instead, we can use distributed training on a single node with multiple GPU (8 in this example):
   
        bash tools/dist_train.sh configs/dery/imagenet/10m_imagenet_128x8_100e_dery_adamw_freeze.py 8
   
   Refer to [mmclassification](https://mmclassification.readthedocs.io/en/1.x/user_guides/train_test.html) for more
   documentation on model training and configs, including instructions for distributing across multiple nodes.

 
## 🚛 Other Resources
1. We use several pre-trained models not included in timm and mmcls, listed in [Pre-trained](assets/pre-trained.md).

## Thanks for the support
[![Stargazers repo roster for @Adamdad/DeRy](https://reporoster.com/stars/Adamdad/DeRy)](https://github.com/Adamdad/DeRy/stargazers)

## ✍ Citation

```bibtex
@article{yang2022dery,
    author    = {Xingyi Yang, Daquan Zhou, Songhua Liu, Jingwen Ye, Xinchao Wang},
    title     = {Deep Model Reassembly},
    journal   = {NeurIPS},
    year      = {2022},
}
```
# Extensions
1. Extension on **Efficient and Parallel Training**: [Deep-Incubation](https://arxiv.org/abs/2212.04129)
2. Extension for **Efficient Model Zoo Training**:  [Stitchable Neural Networks](https://arxiv.org/abs/2302.06586)

