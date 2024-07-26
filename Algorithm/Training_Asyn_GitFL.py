import numpy as np
import copy
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.generator import Generator
from models.Update import LocalUpdate_FedAvg,DatasetSplit
from models.Fed import Aggregation
from models.test import test_img
from utils.utils import save_result
import utils.asynchronous_client_config as AsynConfig


def GitFL(args, net_glob, dataset_train, dataset_test, dict_users):
    # initialize
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
    
    acc = []
    times = []
    num_model = int(args.frac * args.num_users)


    buffer_weights = []
    wait_merge_phy_times = []
    wait_merge_weights = []
    trained_model_phy_times = []
    wait_merge_clients = []
    model_physical_times = []
    model_comm_times = []
    
    for i in range(num_model):
        weight = copy.deepcopy(net_glob.state_dict())
        buffer_weights.append(weight)
        wait_merge_weights.append(weight)
        model_physical_times.append(0.0)
        wait_merge_phy_times.append(0.0)
        trained_model_phy_times.append(0.0)
        wait_merge_clients.append(-1)
        model_comm_times.append(0)
    
    client_comm_time_table = [0 for _ in range(args.num_users)]
    client_physical_time_table = [0 for _ in range(args.num_users)]

    physical_time_bound = args.physical_time
    comm_time_bound = args.comm_time

    physical_time = 0.0
    comm_time = 0.0
    isNext = True

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
        
        train_model_idx = 0
        min_physical_time = wait_merge_phy_times[0]
        for m_i in range(num_model):
            if wait_merge_phy_times[m_i] < min_physical_time:
                min_physical_time = wait_merge_phy_times[m_i]
                train_model_idx = m_i
        wait_merge_client = wait_merge_clients[train_model_idx]
        physical_time = min_physical_time
        if wait_merge_client != -1:
            client_comm = client_comm_time_table[wait_merge_client]
            client_phy_time = trained_model_phy_times[train_model_idx]
            client_physical_time_table[wait_merge_client] = (client_physical_time_table[wait_merge_client] * (client_comm - 1) + client_phy_time)/(client_comm)
        comm_time += 1
        model_comm_times[train_model_idx] += 1

        user_idx = client_select(args, physical_time, train_model_idx, model_comm_times, model_physical_times, client_comm_time_table, client_physical_time_table, wait_merge_clients)

        buffer_weights[train_model_idx] = copy.deepcopy(wait_merge_weights[train_model_idx])

        agg_control = model_comm_times[train_model_idx] - sum(model_comm_times)/len(model_comm_times)

        train_weight = main_model_generation(buffer_weights,model_comm_times)
        train_weight = main_model_generation([wait_merge_weights[train_model_idx],train_weight],[min(50.0,max(2.0, agg_control + 10.0)) ,1.0])

        model_physical_times[train_model_idx] = wait_merge_phy_times[train_model_idx]
        run_time = asyn_clients[user_idx].get_train_time() + asyn_clients[user_idx].get_comm_time()
        wait_merge_phy_times[train_model_idx] += run_time
        trained_model_phy_times[train_model_idx] = run_time
        wait_merge_clients[train_model_idx] = user_idx
        
        client_comm_time_table[user_idx] += 1
        # train
        net_glob.load_state_dict(train_weight)

        local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[user_idx])

        w = local.train(net=copy.deepcopy(net_glob).to(args.device))
        
        wait_merge_weights[train_model_idx] = (copy.deepcopy(w))

        if comm_time % 50 == 49:
            net_glob.load_state_dict(main_model_generation(buffer_weights,model_comm_times))
            acc.append(test(net_glob, dataset_test, physical_time, comm_time, args))
            times.append(physical_time)

    save_result(acc, 'test_acc_{}'.format(args.gitfl_select_ctrl), args)
    save_result(times, 'test_time_{}'.format(args.gitfl_select_ctrl), args)

def client_select(args, physical_time, model_idx, model_comm_times, model_physical_times, client_comm_time_table, client_physical_time_table, wait_merge_clients = []):
    select_ctrl = args.gitfl_select_ctrl
    if select_ctrl == -1:
        idx = random.randint(0,args.num_users-1)
        while idx in wait_merge_clients:
            idx = random.randint(0,args.num_users-1)
        return idx
    else:
        weight_table = {}
        avg_comm = 0.0
        avg_time = 0.0
        max_time = 0.0
        for comm in model_comm_times:
            avg_comm += comm
        for phy_time in client_physical_time_table:
            avg_time += phy_time
            max_time = max_time if phy_time < max_time else phy_time
        avg_time /= len(client_physical_time_table)
        avg_comm = avg_comm/len(model_comm_times)
        model_comm = model_comm_times[model_idx]
        comm_ctrl = model_comm - avg_comm
        # if comm_ctrl > 0:
        #     comm_ctrl = comm_ctrl **2
        # else:
        #     comm_ctrl = - ((-comm_ctrl) **2)
        for i in range(len(client_comm_time_table)):
            comm_time = client_comm_time_table[i]
            curiosity = 1.0 / ((comm_time+1.0)**(0.5))
            if max_time == 0.0:
                time_ctrl = 0
            else:
                time_ctrl = ((client_physical_time_table[i] - avg_time))/(max_time*10) * comm_ctrl
            if select_ctrl == 1:
                weight = max(0.000001, time_ctrl)
            elif select_ctrl == 2:
                weight = max(0.000001, curiosity)
            else:
                alpha = physical_time/(args.physical_time*2)
                if physical_time < args.physical_time/10:
                    weight = max(0.000001, curiosity)
                else:
                    weight = max(0.000001, curiosity*alpha + time_ctrl*(1-alpha))
            if weight != 0 and i  not in wait_merge_clients:
                weight_table.setdefault(i,weight)
        return random_weight(weight_table)

def random_weight(weight_table):
    sum_weight = sum(weight_table.values())
    ra = random.uniform(0.0, sum_weight)
    sub_sum = 0.0
    result = 0
    for k in weight_table.keys():
        sub_sum += weight_table[k]
        if ra <= sub_sum:
            result = k
            break
    return result

def main_model_generation(w, lens):
    w_avg = None
    total_count = sum(lens)

    for i in range(0, len(w)):
        if i == 0:
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k] * lens[i]
        else:
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * lens[i]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg

def test(net_glob, dataset_test, phy_time, comm_time, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Physical time: {:.2f}".format(phy_time))
    print("Communication time: {:.2f}".format(comm_time))
    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()