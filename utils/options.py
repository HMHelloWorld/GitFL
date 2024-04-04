#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=2000, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument("--algorithm", type=str, default="FedDC")
    parser.add_argument("--limit_time", type=int, default=300000)

    # model arguments
    parser.add_argument('--model', type=str, default='resnet20', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--generate_data', type=int, default=0, help="whether generate new dataset")
    parser.add_argument('--iid', type=int, default=1, help='whether i.i.d or not')
    parser.add_argument('--noniid_case', type=int, default=0, help="non i.i.d case (1, 2, 3, 4)")
    parser.add_argument('--data_beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--prox_alpha', type=float, default=0.01, help='The hypter parameter for the FedProx')
    parser.add_argument('--lr_g', type=float, default=0.1, help="global learning rate for SCAFFOLD")
    parser.add_argument('--contrastive_alpha', type=float, default=5, help='The hypter parameter for the Moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--ensemble_alpha', type=float, default=0.2, help='The hypter parameter for the FedGKD')
    parser.add_argument('--sim_type', type=str, default='L1', help='Cluster Sampling: cosine or L1 or L2')
    parser.add_argument('--alpha_coef', type=float, default=1e-2, help='FedDC')
    parser.add_argument('--p', type=float, default=2.0, help='power for AT')
    parser.add_argument('--trans_beta', type=int, default=500, help='beta of FedAttTrans')

    # FedASync
    parser.add_argument('--FedASync_alpha', type=float, default=0.9, help='The hypter parameter for the FedASync')
    parser.add_argument('--poly_a', type=float, default=0.5, help='The hypter parameter for the FedASync_S')
    # parser.add_argument('--max_staleness', type=float, default=4, help="")

    # SAFA
    parser.add_argument('--P_frac', type=float, default=0.5)
    parser.add_argument('--max_tolerate', type=int, default=5)
    parser.add_argument('--limit', type=int, default=250)

    # FedSA
    parser.add_argument('--M_frac', type=float, default=0.5)

    # FedMut
    parser.add_argument('--radius', type=float, default=4.0)
    parser.add_argument('--min_radius', type=float, default=0.1)
    parser.add_argument('--mut_acc_rate', type=float, default=0.3)
    parser.add_argument('--mut_bound', type=int, default=50)

     # asynchronous arguments
    parser.add_argument('--asyn_type', type=int, default=1, help='0 communication time, 1 physical time')
    parser.add_argument('--comm_time', type=int, default=10000, help='bound with communication time')
    parser.add_argument('--physical_time', type=float, default=300000, help='bound with physical time')
    parser.add_argument('--uncertain_type', type=int, default=0, help='0 [0.2,0.2,0.2,0.2,0.2]; 1 [0.5,0.2,0.1,0.1,0.1]; 2 [0.1,0.15,0.5,0.15,0.1]; 3 [0.1,0.1,0.1,0.2,0.5].')
    
    parser.add_argument('--fedmerge_select_ctrl', type=int, default=0, help='-1 random, 1 without curiosity, 2 without time ctrl')


    args = parser.parse_args()
    return args
