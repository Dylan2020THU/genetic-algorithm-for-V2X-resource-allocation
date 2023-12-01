import os
import numpy as np
import env_v2x
import time
import sys
import torch
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定显卡1，即第二张


up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]

width = 750
height = 1299

current_dir = os.path.dirname(os.path.realpath(__file__))
label = 'marl_model'


n_veh = int(60)

agent_model = 'veh60_nb3_pkg2_RB20_2023-10-24-13-13-31'

# agent_file = current_dir + "/" + label +  "/" + agent_model
agent_file = "/mnt/HDD/zhangxuan/TVT_marl_model/MFMARL/" + agent_model  # 硬盘上的模型

n_neighbor = 3  # Unicast:1 Broadcast:>1
n_adjacent = n_veh-1  # the number of adjacent neighbors
# n_RB = 8  # 8个子带，高难度
n_RB = 20  # 20个子带，高难度
n_power = 4
# n_pkg = int(sys.argv[3])
n_pkg = 2  # 1060*2 Bytes


# 车速输入
velocity_min = 10  # 车速要输入！
velocity_interval = 5   # 原文代码的速度差就是 ΔV = 5 m/s
velocity_range = np.random.randint(velocity_min, velocity_min + velocity_interval)  # 这个要输入到环境中


IS_TRAIN = 0
IS_TEST = 1 - IS_TRAIN


lambda_V2I = 0.1
env = env_v2x.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_RB, n_neighbor, n_pkg, lambda_V2I)

env.new_random_game()  # initialize parameters in env

n_episode = 500
n_step_per_episode = int(env.time_slow / env.time_fast)  # 0.1/0.001 = 100
# 测试的step最好和训练的step一致
epsi_final = 0.02


def get_state(env, idx=(0, 0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """
    vehicle_trans_idx = idx[0]
    vehicle_recieve_idx = env.vehicles[idx[0]].destinations[idx[1]]
    
    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10) / 35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60  # 这个其实是比较全局的观测
    V2V_channel_local = (env.V2V_channels_with_fastfading[idx[0], env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    # V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :]
                # - np.repeat(env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]][:, np.newaxis], env.n_RB, axis=1) + 10) / 35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
    # print(V2V_interference)
    
    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0  # size: 60*1  # 这个应该可以去掉

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate(((vehicle_trans_idx, vehicle_recieve_idx), V2I_fast, np.reshape(V2V_channel_local, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs,
                           time_remaining, load_remaining, np.asarray([ind_episode, epsi])))  # local obs

def get_remain_B_T(env, idx=(0, 0)):
    """ Get remaining B from the environment """

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return load_remaining, time_remaining

print(len(get_state(env)))



# velocity_range = np.random.randint(velocity_min, velocity_min + velocity_interval)  # 把车速输入到环境中
env.new_random_game()  # initialize parameters in env 初始环境

record_V2V_success = np.zeros([n_step_per_episode, 1])
record_V2I_Rate = np.zeros((n_step_per_episode, n_RB))
record_V2V_Rate = np.zeros((n_step_per_episode, n_veh*n_neighbor))
record_B = np.zeros((n_step_per_episode, n_veh*n_neighbor))  # 全都用0初始化，然后后面要 +=

# 【修改】
# 自己加一个：传输时间的统计
# 100r*n_vehc，每行对应一个episode的每个agent的传输完成时间，每列对应每个agent的传输完成时间
# 行：episode，列：agent
record_trans_time_all_agent = np.zeros((n_episode, n_veh)) + 300  # 起始：300，因为传输时间可能是0

# 最后面，对每一行求max，得出的就是每个episode的传输时间record_trans_time
record_trans_time_per_episode = np.zeros((n_episode, 1)) + 300 # 起始：-1
# 【此处修改完毕，10行以内】

# record_V2V_success_max = np.zeros([n_step_per_episode, n_neighbor])
record_V2V_success_max = np.zeros([n_step_per_episode, 1])
record_V2I_Rate_max = np.zeros((n_step_per_episode, n_RB))
# record_V2V_Rate_max = np.zeros((n_step_per_episode, n_veh))
record_V2V_Rate_max = np.zeros((n_step_per_episode, n_veh*n_neighbor))
record_B_max = np.zeros((n_step_per_episode, n_veh*n_neighbor))

# record_V2V_success_min = np.ones([n_step_per_episode, n_neighbor])
record_V2V_success_min = np.ones([n_step_per_episode, 1])
record_V2I_Rate_min = np.ones((n_step_per_episode, n_RB))
# record_V2V_Rate_min = np.ones((n_step_per_episode, n_veh))
record_V2V_Rate_min = np.ones((n_step_per_episode, n_veh*n_neighbor))
record_avg_V2V_Rate = np.empty(n_step_per_episode) # TODO
record_B_min = np.ones((n_step_per_episode, n_veh*n_neighbor))
record_reward = np.zeros([n_episode * n_step_per_episode, 1])
record_loss = []
# 功率与能量效率（这组数据暂时有问题）
# 输出功率选择的值（每个step的平均）
# 每一行表示一个step，共100行。每列是一个agent。表格的尺寸和
# 在物理学里面，1 W = 1 J/s，最后面能量效率的单位大概是 bits / J之类的
# 参考肖海林的文献
record_V2V_power = np.zeros((n_episode*n_step_per_episode, n_veh))  # in mW
V2V_power_dB_List = [23, 15, 5, -100]  # the power levels in dBm
# V2V_power_List = 10 ** (V2V_power_dB_List / 10)  # in mW
V2V_power_List = [10**(23/10), 10**(15/10), 10**(5/10), 0]  # in mW

# 计算公式：V2V能量效率 = np.sum(record_V2V_Rate) / np.sum(record_V2V_power)

if IS_TEST:
    ''' 1 Initialize: To randomly generate all agents' actions '''
    a_spec = np.random.randint(low=0, high=n_RB,
                            size=n_veh)  # To randomly generate sub-band and power for each agent. Total: n_veh*2)
    a_power = np.random.randint(low=0, high=n_power, size=n_veh)
    a = np.concatenate((a_spec.reshape(1, -1), a_power.reshape(1, -1)), axis=0)  # concatenate according to row
    action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    for i in range(n_veh):
        for j in range(n_neighbor):
            action_all_training[i, j, 0] = a[0, i]
            action_all_training[i, j, 1] = a[1, i] # Joint action

    ''' 2 Genetic Algorithm'''
    for i_episode in range(n_episode):  # 第一个tab：回合
        print("-------------------------")
        print('Episode:', i_episode)

        # Update the environment
        env.renew_positions()
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()
        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        for i_step in range(n_step_per_episode):
            # 2.1 calculate the fitness of each agent
            time_step = i_episode * n_step_per_episode + i_step
            for i in range(n_veh):
                for j in range(n_neighbor):  # the number of neighbor to deliver message
                    record_V2V_power[time_step, i] = V2V_power_List[action_all_training[i, j, 1]]

            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()  # To take joint action a
            # train_reward = env.act_for_training(action_temp)
            test_V2I_Rate, test_V2V_success, test_V2V_Rate = env.act_for_testing(
                action_temp)  # To obtain the reward with joint action a
            record_V2V_success[i_step] += test_V2V_success  # The reward of each time step
            if test_V2V_success > record_V2V_success_max[i_step]:
                record_V2V_success_max[i_step] = test_V2V_success
            if test_V2V_success < record_V2V_success_min[i_step]:
                record_V2V_success_min[i_step] = test_V2V_success

            record_V2I_Rate[i_step] += test_V2I_Rate  # The reward of each time step
            for index_V2I_Rate in range(len(record_V2I_Rate[i_step])):
                if test_V2I_Rate[index_V2I_Rate] > record_V2I_Rate_max[i_step][index_V2I_Rate]:
                    record_V2I_Rate_max[i_step][index_V2I_Rate] = test_V2I_Rate[index_V2I_Rate]
                if test_V2I_Rate[index_V2I_Rate] < record_V2I_Rate_min[i_step][index_V2I_Rate]:
                    record_V2I_Rate_min[i_step][index_V2I_Rate] = test_V2I_Rate[index_V2I_Rate]

            tmp = np.squeeze(test_V2V_Rate).reshape(1, -1)
            record_V2V_Rate[i_step] += tmp[0]  # The reward of each time step
            for index_V2V_Rate in range(len(record_V2V_Rate[i_step])):
                if tmp[0, index_V2V_Rate] > record_V2V_Rate_max[i_step][index_V2V_Rate]:
                    record_V2V_Rate_max[i_step][index_V2V_Rate] = record_V2V_Rate[i_step][index_V2V_Rate]
                if tmp[0, index_V2V_Rate] < record_V2V_Rate_min[i_step][index_V2V_Rate]:
                    record_V2V_Rate_min[i_step][index_V2V_Rate] = record_V2V_Rate[i_step][index_V2V_Rate]
            # TODO: how to calculate the fitness of each agent?

            # calculate the average V2V rate
            record_avg_V2V_Rate[i_step] /= n_veh
            fitness = record_avg_V2V_Rate # TODO

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            for i in range(n_veh):
                for j in range(n_neighbor):
                    # state
                    remain_B = 0
                    remain_T = 10e5
                    remain_B_tmp, remain_T_tmp = get_remain_B_T(env, [i, j])
                    remain_B += remain_B_tmp
                    if remain_T_tmp < remain_T:
                        remain_T = remain_T_tmp

                    record_B[i_step, n_neighbor * i + j] += remain_B
                    if remain_B < record_B_min[i_step, n_neighbor * i + j]:
                        record_B_min[i_step, n_neighbor * i + j] = remain_B
                    if remain_B > record_B_max[i_step, n_neighbor * i + j]:
                        record_B_max[i_step, n_neighbor * i + j] = remain_B

                # 记录传输时间【修改】
                if (remain_B == 0. and record_trans_time_all_agent[i_episode, i] == 300):
                    record_trans_time_all_agent[i_episode, i] = i_step + 1 # 注意要+1，因为这里i_step是for循环的index，从0开始

            # 2.2 select the best 10 chromosomes
            # 2.3 crossover
            # 2.4 mutate
            # 2.5 repeat 2.1-2.4 for 100 times
            # 2.6 return the best chromosome



        '''后面可以删去'''
        trans_time_worst1 = np.max(record_trans_time_all_agent[i_episode, :])
        if trans_time_worst1 > 100:
            print('---------- Transmission unfinished! -----------')
            # break

    ''' 3 RL: Save the test files: V2I rate, V2V rate V2V success, remain B & transmission_time'''
    # 平均传输时间
    record_trans_time_per_episode = np.amax(record_trans_time_all_agent, axis=1)  # 对行求最大值，得到每个回合的传输完成时间
    trans_time_worst = np.max(record_trans_time_per_episode)
    if trans_time_worst == 300:
        print('---------- Transmission unfinished! -----------')
        # continue

    # data_path_dir = current_dir + "/" + "model/" + label + "/" + sys.argv[1] + "_test_env"
    data_path_dir = current_dir + "/" + "model/" + label + "/" + agent_model + f"_Payload{n_pkg}_TestEpi{n_episode}" + "_test_env"
    folder = os.path.exists(data_path_dir)
    if not folder:
        os.makedirs(data_path_dir)
        print("make dir " + data_path_dir + " successful!")
    else:
        print("There is a folder named " + data_path_dir + "!")


    # V2V success
    record_V2V_success /= n_episode  # 这里相当于求平均，求和值/n_episode
    # debug发现：这个求平均没有问题
    current_dir = os.path.dirname(os.path.realpath(__file__))
    V2V_success_path = data_path_dir + f'/V2V_success_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}_mean.csv'
    np.savetxt(V2V_success_path, record_V2V_success, fmt='%.4f', delimiter=',')
    # record_V2V_success是一个 100行*1列 的数组

    V2V_success_path = data_path_dir + f'/V2V_success_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}_min.csv'
    np.savetxt(V2V_success_path, record_V2V_success_min, fmt='%.4f', delimiter=',')

    V2V_success_path = data_path_dir + f'/V2V_success_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}_max.csv'
    np.savetxt(V2V_success_path, record_V2V_success_max, fmt='%.4f', delimiter=',')

    # V2I Rate every episode(mean)
    record_V2I_Rate /= n_episode
    V2I_Rate_path = data_path_dir +  f'/V2I_Rate_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}_mean.csv'
    np.savetxt(V2I_Rate_path, record_V2I_Rate, fmt='%.4f', delimiter=',')

    # mean(sum(V2I Rate)) 平均的V2I总速率
    mean_sum_V2I_rate = np.sum(record_V2I_Rate) / n_step_per_episode
    mean_sum_V2I_rate = np.array([mean_sum_V2I_rate])

    mean_sum_V2I_rate_path = data_path_dir + f'/mean_sum_V2I_rate_age{n_veh}_rb{n_RB}_pow{n_power}.txt'
    np.savetxt(mean_sum_V2I_rate_path, mean_sum_V2I_rate, fmt='%.4f')

    # V2V Rate
    record_V2V_Rate /= n_episode
    V2V_Rate_path = data_path_dir +  f'/V2V_Rate_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}_mean.csv'
    np.savetxt(V2V_Rate_path, record_V2V_Rate, fmt='%.4f', delimiter=',')

    # Remain B
    record_B /= n_episode
    Remain_B_path = data_path_dir + f'/remain_B_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}_mean.csv'
    np.savetxt(Remain_B_path, record_B, fmt='%.4f', delimiter=',')

    Remain_B_path = data_path_dir + f'/remain_B_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}_min.csv'
    np.savetxt(Remain_B_path, record_B_min, fmt='%.4f', delimiter=',')

    Remain_B_path = data_path_dir + f'/remain_B_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}_max.csv'
    np.savetxt(Remain_B_path, record_B_max, fmt='%.4f', delimiter=',')

    '''丢包率'''
    # 自己增加的
    packet_loss_rate = np.mean(record_B[99, :])
    packet_loss_rate = np.array([packet_loss_rate])
    packet_loss_rate_path = data_path_dir + f'/packet_loss_rate_age{n_veh}_rb{n_RB}_pow{n_power}.txt'
    np.savetxt(packet_loss_rate_path, packet_loss_rate, fmt='%.4f')

    # 传输时间
    # [修改]自己增加的
    record_trans_time_per_episode = np.amax(record_trans_time_all_agent, axis=1)  # 对行求最大值，得到每个回合的传输完成时间

    # 每个回合的传输时间 trans_time
    # 如果是控制变量法做实验的，要记得把变量再加到文件名上
    trans_time_per_episode_path = data_path_dir + f'/trans_time_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}.csv'
    np.savetxt(trans_time_per_episode_path, record_trans_time_per_episode, fmt='%.4f', delimiter=',')

    # 所有agent的传输时间
    trans_time_all_agent_path = data_path_dir + f'/trans_time_all_agent_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}.csv'
    np.savetxt(trans_time_all_agent_path, record_trans_time_all_agent, fmt='%.4f', delimiter=',')

    # 求传输时间的平均值、最大值、最小值
    record_trans_time_best_mean_worst = np.zeros((3, 1)) + 300  # 生成变量，预分配空间
    record_trans_time_best_mean_worst[0, 0] = np.min(record_trans_time_per_episode)  # 对列求最小值，最快传输时间
    record_trans_time_best_mean_worst[1, 0] = np.mean(record_trans_time_per_episode)  # 求均值，平均时间
    record_trans_time_best_mean_worst[2, 0] = np.max(record_trans_time_per_episode)  # 求最大值，最差情况，最长时间
    # 保存文件
    trans_time_best_mean_worst = data_path_dir + f'/trans_time_best_mean_worst_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}.csv'
    np.savetxt(trans_time_best_mean_worst, record_trans_time_best_mean_worst, fmt='%.4f', delimiter=',')

        # # V2V power
    # V2V_power_path = data_path_dir + f'/V2V_power_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}_rs{rs}.csv'
    # np.savetxt(V2V_power_path, record_V2V_power, fmt='%.4f', delimiter=',')

    # V2V power 选择概率
    times_23dBm = np.sum(record_V2V_power == V2V_power_List[0])
    times_15dBm = np.sum(record_V2V_power == V2V_power_List[1])
    times_5dBm = np.sum(record_V2V_power == V2V_power_List[2])
    times_0dBm = np.sum(record_V2V_power == V2V_power_List[3])
    # 计算概率
    # 传输时间步的总数
    sum_trans_time = times_23dBm + times_15dBm + times_5dBm
    probability_23dBm = times_23dBm / sum_trans_time  # 选择23dBm的概率
    probability_15dBm = times_15dBm / sum_trans_time  # 选择15dBm的概率
    probability_5dBm = times_5dBm / sum_trans_time  # 选择5dBm的概率
    record_P_23_15_5dBm = np.array([probability_23dBm, probability_15dBm, probability_5dBm])
    # 保存文件
    V2V_power_Selection_P_path = data_path_dir + f'/V2V_power_Selection_P_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}.csv'
    np.savetxt(V2V_power_Selection_P_path, record_P_23_15_5dBm, fmt='%.4f', delimiter=',')

    # V2V能源效率
    # 计算公式：V2V能量效率 = np.sum(record_V2V_Rate) / np.sum(record_V2V_power)
    V2V_EE = np.sum(record_V2V_Rate) / np.sum(record_V2V_power) * int(1e3)  # V2V 能量效率, in bits/J
    V2V_EE = np.array([V2V_EE])
    # 保存文件
    V2V_EE_path = data_path_dir + f'/V2V_EE_bits_J_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}.txt'
    np.savetxt(V2V_EE_path, V2V_EE, fmt='%.4f')

    # 计算平均频谱效率并输出
    payload_real = 1060 * n_pkg  # in Bytes
    payload_bits = payload_real * 8 * n_veh # in bits，分子项
    W = 1e6
    avg_t = np.mean(record_trans_time_all_agent)*1e-3  # 平均传输时间  in s
    denominator = W*avg_t*n_RB  # 分母项
    tau = payload_bits / denominator  # in bits/s/Hz
    tau = np.array([tau])
    # 保存文件
    avg_sepctrum_efficiency_path = data_path_dir + f'/avg_sepctrum_efficiency_age{n_veh}_rb{n_RB}_pow{n_power}_Payload{n_pkg}.txt'
    np.savetxt(avg_sepctrum_efficiency_path, tau, fmt='%.4f')

# 程序结束
print('------------Test Finished------------')
