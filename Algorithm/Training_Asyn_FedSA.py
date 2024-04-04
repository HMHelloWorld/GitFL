from models.Fed import *
from models.Update import LocalUpdate_FedSA
from utils.Clients import Clients
from utils.utils import *
from models.test import test_img


def FedSA(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()
    acc = []
    time_list = []
    comm_list = []
    comm_count = 0

    select_list = [1 for _ in range(args.num_users)]
    clients = Clients(args)
    start_time = 0

    local_result = [0 for _ in range(args.num_users)]

    M = int(args.M_frac * (args.num_users * args.frac + 1))
    max_tolerate = 100

    lens = []
    for idx in range(args.num_users):
        lens.append(len(dict_users[idx]))
    
    physical_time_bound = args.physical_time
    comm_time_bound = args.comm_time
    physical_time = 0.0
    comm_time = 0.0
    isNext = True

    iter = 0

    while isNext:
        # print (model_comm_times)
        # print (wait_merge_clients)
        # print (wait_merge_phy_times)
        if args.asyn_type == 0:
            if comm_time >= comm_time_bound:
                isNext = False
        else:
            if physical_time >= physical_time_bound:
                isNext = False
        iter += 1
        print('*' * 80)
        print("start_time:", start_time)

        outdated = set()
        if iter == 1:
            train = clients.get_idle(int(args.num_users * args.frac))
        else:
            train = clients.get_idle(M)
        count = 0
        for idx in train:
            clients.train(idx, iter - 1)
            count += 0.5
            comm_count += 0.5
            
        print(train)
        for idx, version, time in clients.update_list:
            client = clients.get(idx)
            if client.version < iter - max_tolerate:
                clients.train(idx, iter - 1)
                count += 1
                comm_count += 0.5
                comm_time += 0.5
                train.append(idx)

        # for idx in range(args.num_users):
        #     client = clients.get(idx)
        #     if client.version == iter - 1 or client.version < iter - args.max_tolerate:
        #         train.append(idx)
        #         count += 1
        #         comm_count += 0.5
        #         if client.version < iter - args.max_tolerate:
        #             outdated.add(idx)
        #         clients.train(idx, iter - 1)
        for idx in train:
            select_list[idx] += 1
        # print("train:", train)
        # print("train_num:", count)
        # print(clients.update_list)
        lens = {}
        for idx, version, time in clients.update_list:
            lens[idx] = len(dict_users[idx])
        update_users = clients.pop_update(M)
        update_w = {}
        for idx, version, time in update_users:
            local = LocalUpdate_FedSA(args=args, dataset=dataset_train, idxs=dict_users[idx])
            lr = args.lr / (args.num_users * (select_list[idx] / sum(select_list)))
            w = local.train(net=copy.deepcopy(net_glob).to(args.device), lr=lr)
            # for key in w.keys():
            #     w[key] = w[key].cpu()
            # local_result[idx] = w
            update_w[idx] = w

        comm_count += 0.5 * M
        start_time += update_users[-1][2]

        w_glob = copy.deepcopy(net_glob).state_dict()
        w_glob = Weighted_Aggregation_FedSA(update_w, lens, w_glob)

        net_glob.load_state_dict(w_glob)

        for idx, version, time in update_users:
            clients.get(idx).version = iter

        # M = estimate_M()

        acc.append(test(net_glob, dataset_test, args))
        physical_time = start_time
        time_list.append(start_time)
        comm_list.append(comm_count)

    save_result(acc, 'test_acc{}'.format(args.uncertain_type), args)
    save_result(time_list, 'test_time{}'.format(args.uncertain_type), args)
    save_result(comm_list, 'test_comm{}'.format(args.uncertain_type), args)


def estimate_M():
    pass


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()
