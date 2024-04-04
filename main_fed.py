#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import copy

from utils.options import args_parser
from utils.set_seed import set_random_seed
from models.Update import *
from models.Nets import *
from models.MobileNetV2 import MobileNetV2
from models.Fed import Aggregation, Weighted_Aggregation_FedASync
from models.test import test_img
from models.resnetcifar import *
from models import *
from utils.get_dataset import get_dataset
from utils.utils import save_result,save_model
from Algorithm.Training_Asyn_FedSA import FedSA
from Algorithm.Training_Asyn_GitFL import GitFL
from utils.Clients import Clients
import utils.asynchronous_client_config as AsynConfig

def FedAvg(net_glob, dataset_train, dataset_test, dict_users):

    net_glob.train()

    uncertain_list = [0.2,0.2,0.2,0.2,0.2]
    if args.uncertain_type == 1:
        uncertain_list = [0.5,0.2,0.1,0.1,0.1]
    elif args.uncertain_type == 2:
        uncertain_list = [0.1,0.15,0.5,0.15,0.1]
    elif args.uncertain_type == 3:
        uncertain_list = [0.1,0.1,0.1,0.2,0.5]
    elif args.uncertain_type == 4:
        uncertain_list = [0.4,0.1,0.0,0.1,0.4]

    asyn_clients = AsynConfig.generate_asyn_clients(uncertain_list,uncertain_list,args.num_users)

    times = []
    total_time = 0

    # training
    acc = []
    loss = []
    train_loss=[]

    for iter in range(args.epochs):

        print('*'*80)
        print('Round {:3d}'.format(iter))


        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        max_time = 0
        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            run_time = asyn_clients[idx].get_train_time() + asyn_clients[idx].get_comm_time()
            if max_time < run_time:
                max_time = run_time
        total_time += max_time
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        if iter % 10 == 9:
            item_acc,item_loss = test_with_loss(net_glob, dataset_test, args)
            ta,tl = test_with_loss(net_glob, dataset_train, args)
            acc.append(item_acc)
            loss.append(item_loss)
            train_loss.append(tl)
            times.append(total_time)

    save_result(acc, 'test_acc', args)
    save_result(loss, 'test_loss', args)
    save_result(times, 'test_time', args)
    save_result(train_loss, 'test_train_loss', args)
    save_model(net_glob.state_dict(), 'test_model', args)


def FedProx(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)

from utils.clustering import *
from scipy.cluster.hierarchy import linkage
def ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users):

    net_glob.to('cpu')

    n_samples = np.array([len(dict_users[idx]) for idx in dict_users.keys()])
    weights = n_samples / np.sum(n_samples)
    n_sampled = max(int(args.frac * args.num_users), 1)

    gradients = get_gradients('', net_glob, [net_glob] * len(dict_users))

    net_glob.train()

    # training
    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        previous_global_model = copy.deepcopy(net_glob)
        clients_models = []
        sampled_clients_for_grad = []

        # GET THE CLIENTS' SIMILARITY MATRIX
        if iter == 0:
            sim_matrix = get_matrix_similarity_from_grads(
                gradients, distance_type=args.sim_type
            )

        # GET THE DENDROGRAM TREE ASSOCIATED
        linkage_matrix = linkage(sim_matrix, "ward")

        distri_clusters = get_clusters_with_alg2(
            linkage_matrix, n_sampled, weights
        )

        w_locals = []
        lens = []
        idxs_users = sample_clients(distri_clusters)
        for idx in idxs_users:
            local = LocalUpdate_ClientSampling(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_model = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local_model.to('cpu')

            w_locals.append(copy.deepcopy(local_model.state_dict()))
            lens.append(len(dict_users[idx]))

            clients_models.append(copy.deepcopy(local_model))
            sampled_clients_for_grad.append(idx)

            del local_model
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        gradients_i = get_gradients(
            '', previous_global_model, clients_models
        )
        for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
            gradients[idx] = gradient

        sim_matrix = get_matrix_similarity_from_grads_new(
            gradients, distance_type=args.sim_type, idx=idxs_users, metric_matrix=sim_matrix
        )

        net_glob.to(args.device)
        acc.append(test(net_glob, dataset_test, args))
        net_glob.to('cpu')

        del clients_models

    save_result(acc, 'test_acc', args)

def FedASync(args, net_glob, dataset_train, dataset_test, dict_users):

    net_glob.train()

    acc = []
    time_list = []

    clients = Clients(args)
    start_time = 0
    m = max(int(args.frac * args.num_users), 1)
    local_result = [None for _ in range(args.num_users)]

    for iter in range(args.epochs):
        if start_time > args.limit_time:
            break

        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)
        if iter == 0:
            idxs_users = clients.get_idle(m)
            if len(idxs_users):
                for idx in idxs_users:
                    local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train,
                                                idxs=dict_users[idx])
                    w = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    w_local = copy.deepcopy(w)
                    local_result[idx] = w_local
                    clients.train(idx, iter)

        update_idx, version, time = clients.pop_update(1)[0]
        lag = iter - version
        start_time += time

        alpha = args.FedASync_alpha * ((lag + 1) ** -args.poly_a)

        w_new = copy.deepcopy(net_glob).to(args.device).state_dict()
        w_new = Weighted_Aggregation_FedASync(local_result[update_idx], w_new, alpha)
        net_glob.load_state_dict(w_new)
        local_result[update_idx] = None

        acc.append(test(net_glob, dataset_test, args))
        time_list.append(start_time)

        idx = clients.get_idle(1)[0]
        local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train,
                                    idxs=dict_users[idx])
        w = local.train(net=copy.deepcopy(net_glob).to(args.device))
        w_local = copy.deepcopy(w)
        local_result[idx] = w_local
        clients.train(idx, iter)

    save_result(acc, 'test_acc{}'.format(args.uncertain_type), args)
    save_result(time_list, 'test_time{}'.format(args.uncertain_type), args)
    # save_result(comm_list, 'test_comm{}'.format(args.uncertain_type), args)


def SAFA(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.to('cpu')
    net_glob.train()
    acc = []
    time_list = []
    comm_list = []
    comm_count = 0

    clients = Clients(args)
    start_time = 0

    cache = [copy.deepcopy(net_glob.state_dict()) for i in range(args.num_users)]
    local_result = [0 for _ in range(args.num_users)]
    net_glob.to(args.device)

    pre_P = set()
    P = set()
    next_num = 10


    physical_time_bound = args.physical_time
    comm_time_bound = args.comm_time
    physical_time = 0.0
    comm_time = 0.0
    isNext = True

    iter = 0

    while isNext:
        if args.asyn_type == 0:
            if comm_time >= comm_time_bound:
                isNext = False
        else:
            if physical_time >= physical_time_bound:
                isNext = False
        iter += 1
        print('*' * 80)
        print('Round {:3d}'.format(iter))
        print("start_time:", start_time)

        outdated = set()
        count = 0
        train = clients.get_idle(next_num)
        for idx in train:
            clients.train(idx, iter - 1)
            count += 1
            comm_count += 0.5
        for idx, version, time in clients.update_list:
            client = clients.get(idx)
            if client.version < iter - args.max_tolerate:
                clients.train(idx, iter - 1)
                count += 1
                comm_count += 0.5
                train.append(idx)

        update_users = clients.get_update_byLimit(args.limit)

        Q = []
        count = 0
        for idx, version, time in update_users:
            if len(P) == 10 * args.P_frac:
                break
            if idx not in pre_P:
                P.add(idx)
            else:
                Q.append(idx)
            count += 1
        update_users = update_users[0:count]

        for idx, version, time in update_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device))
            for key in w.keys():
                w[key] = w[key].cpu()
            local_result[idx] = w

        update_list = clients.update_list[::]
        clients.pop_update(count)
        comm_count += 0.5 * count
        comm_time = comm_count
        start_time += min(args.limit, update_users[-1][2])
        physical_time = start_time
        next_num = count

        if len(P) < args.num_users * args.frac:
            q = min(int(args.num_users * args.frac - len(P)), len(Q))
            P = P.union(set(Q[0:q]))
            Q = Q[q::]

        # print("P:", P)
        # print("Q:", Q)

        for idx in range(args.num_users):
            if idx in P:
                cache[idx] = local_result[idx]
            elif idx in outdated:
                cache[idx] = copy.deepcopy(net_glob.state_dict())

        c = []
        lens = []
        for idx, version, time in update_list:
            c.append(cache[idx])
            lens.append(len(dict_users[idx]))

        w_glob = Aggregation(c, lens)

        for idx in range(args.num_users):
            if idx in Q:
                cache[idx] = local_result[idx]

        net_glob.load_state_dict(w_glob)

        for idx, version, time in update_users:
            clients.get(idx).version = iter
        pre_P = P
        P = set()

        acc.append(test(net_glob, dataset_test, args))
        time_list.append(start_time)
        comm_list.append(comm_count)

    save_result(acc, 'test_acc{}'.format(args.uncertain_type), args)
    save_result(time_list, 'test_time{}'.format(args.uncertain_type), args)
    save_result(comm_list, 'test_comm{}'.format(args.uncertain_type), args)


def test(net_glob, dataset_test, args):
    
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()

def test_with_loss(net_glob, dataset_test, args):
    
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item(), loss_test

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    set_random_seed(args.seed)

    dataset_train, dataset_test, dict_users = get_dataset(args)

    if args.model == 'cnn' and args.dataset == 'femnist':
        net_glob = CNNFashionMnist(args)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args)
    elif args.use_project_head:
        net_glob = ModelFedCon(args.model, args.out_dim, args.num_classes)
    elif 'cifar' in args.dataset and 'cnn' in args.model:
        net_glob = CNNCifar(args)
    elif args.model == 'resnet20' and args.dataset == 'mnist':
        net_glob = ResNet20_mnist(args=args).to(args.device)
    elif args.model == 'resnet20' and (args.dataset == 'fashion-mnist' or args.dataset == 'femnist'):
        net_glob = ResNet20_mnist(args=args).to(args.device)
    elif args.model == 'resnet20' and args.dataset == 'cifar':
        net_glob = ResNet20_cifar(args=args).to(args.device)
    elif args.model == 'resnet20' and args.dataset == 'cifar100':
        net_glob = ResNet20_cifar(args=args).to(args.device)
    elif 'resnet18' in args.model:
        net_glob = ResNet18_cifar10(num_classes = args.num_classes)
    elif 'resnet50' in args.model:
        net_glob = ResNet50_cifar10(num_classes = args.num_classes)
    elif 'mobilenet' in args.model:
        net_glob = MobileNetV2(args)
    elif 'lstm' in args.model:
        net_glob = CharLSTM()
    elif 'cifar' in args.dataset and args.model == 'vgg':
        net_glob = VGG16(args)
    elif 'mnist' in args.dataset and args.model == 'vgg':
        net_glob = VGG16_mnist(args)

    net_glob.to(args.device)
    print(net_glob)

    if args.algorithm == 'FedAvg':
        FedAvg(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedProx':
        FedProx(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'ClusteredSampling':
        ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedASync':
        FedASync(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'SAFA':
        SAFA(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedSA':
        FedSA(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'GitFL':
        GitFL(args, net_glob, dataset_train, dataset_test, dict_users)


