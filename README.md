# GitFL & Baselines Documentation

GitFL is a novel asynchronous federated learning method that was published in the proceedings of 2023 **IEEE Real-Time Systems Symposium (RTSS 2023)**.
Please see the paper **"GitFL: Uncertainty-Aware Real-Time Asynchronous Federated Learning Using Version Control"** for the details.

https://www.computer.org/csdl/proceedings-article/rtss/2023/285700a145/1UjInWgoXGE


## Environment setting requirements
* Python 3.7
* PyTorch

## Instruction
### 2.1 Parameter
#### 2.1.1 Dataset Setting
    --dataset <dataset name>
We can set ‘cifar10’, ‘cifar100’ and ‘femnist’ for CIFAR-10, CIFAR-100, and FEMNIST.

#### 2.1.2 Model Settings
    --model <model name>
We can set ‘cnn’, ‘resnet18’ and ‘vgg’ for CNN, ResNet-18, and VGG-16.

    --num_classes <number>
Set the number of classes

Set 10 for CIFAR-10

Set 20 for CIFAR-100

Set 62 for FEMNIST

    --num_channels <number>
Set the number of channels of data

Set 3 for CIFAR-10 and CIFAR-100.

Set 1 for FEMNIST.

#### 2.1.3 Data heterogeneity
    --iid <0 or 1>
0 – set non-iid

1 – set iid

    --data_beta <α>
Set the α for the Dirichlet distribution

    --generate_data <0 or 1>
0 – use the existing configuration of Dir(α)

1 – generate a new configuration of Dir(α)

#### 2.1.4 Client Number settings
    --num_users <number>
Set the number of total clients

    --frac <float>
Set the fraction of clients.

E.g. --num_users 100 --frac 0.1 denotes there are 100 clients and 10% of them participant in local training in each FL round.

#### 2.1.5 FL Settings
    --asyn_type <0 or 1>
0 – use the bound of communication time

1 – use the bound of physical time

    --comm_time <number of rounds>
Set the bound of communication time.

    --physical_time <running time>
Set the bound of physical time.

    --uncertain_type <1, 2, 3 or 4>
Set the uncertainty environment.

[execllent, high, medium, low, critical]

0 – [0.2, 0.2, 0.2, 0.2, 0.2]

1 – [0.5, 0.2, 0.1, 0.1, 0.1]

2 – [0.1, 0.15, 0.5, 0.15, 0.1]

3 – [0.1, 0.1, 0.1, 0.2, 0.5]

4 – [0.4, 0.1, 0.0, 0.1, 0.4]


#### 2.1.6 GitFL and Baseline Settings
    --algorithm <baseline name>
Set the baseline name:
GitFL
FedAvg
FedASync
SAFA
FedSA

    --gitfl_select_ctrl <-1, 0, 1, 2>
Set the client selection strategy for GitFL

-1 – Random

0 – Using curiosity and version control

1 – Only using version control

2 – Only using curiosity


### 2.2 Running Project
#### 2.2.1 Basic Commands
    cd GitFL_code/
    python main_fed.py <parameters>

Example
    python main_fed.py --num_users 100 --frac 0.1 --generate_data 1 --num_classes 10 --num_channels 3 --model cnn --algorithm GitFL --dataset cifar10 --iid 0 --data_beta 0.1 --gitfl_select_ctrl 0 --asyn_type 1 --physical_time 100000

    python main_fed.py --num_users 100 --frac 0.1 --generate_data 1 --num_classes 10 --num_channels 3 --model cnn --algorithm GitFL --dataset cifar10 --iid 1 --gitfl_select_ctrl 0 --asyn_type 1 --physical_time 100000

## Citation
```
@inproceedings{hu2023gitfl,
  title={GitFL: Uncertainty-Aware Real-Time Asynchronous Federated Learning Using Version Control},
  author={Hu, Ming and Xia, Zeke and Yan, Dengke and Yue, Zhihao and Xia, Jun and Huang, Yihao and Liu, Yang and Chen, Mingsong},
  booktitle={2023 IEEE Real-Time Systems Symposium (RTSS)},
  pages={145--157},
  year={2023},
  organization={IEEE}
}
```
