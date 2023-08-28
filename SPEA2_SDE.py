#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/20 13:00
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : SPEA2_SDE.py
# @Statement : Strength Pareto evolutionary algorithm 2 with shift-based density estimation (SPEA2-SDE)
# @Reference : Li M, Yang S, Liu X. Shift-based density estimation for Pareto-based algorithms in many-objective optimization[J]. IEEE Transactions on Evolutionary Computation, 2013, 18(3): 348-365.
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def cal_obj(pop, nobj):
    # 0 <= x <= 1
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum((pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))
    objs = np.zeros((pop.shape[0], nobj))
    temp_pop = pop[:, : nobj - 1]
    for i in range(nobj):
        f = 0.5 * (1 + g)
        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)
        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]
        objs[:, i] = f
    return objs


def selection(pop, F, pc, k=2):
    # tournament selection
    (npop, nvar) = pop.shape
    nm = int(npop * pc)  # mating pool size
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, nvar))
    for i in range(nm):
        selections = np.random.choice(npop, k, replace=True)
        ind = selections[np.argmin(F[selections])]
        mating_pool[i] = pop[ind]
    return mating_pool


def crossover(mating_pool, lb, ub, pc, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > pc, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < 1 / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def cal_fitness(objs):
    # calculate the fitness of each individual
    npop = objs.shape[0]
    K = round(np.sqrt(npop))
    S = np.zeros(npop, dtype=int)  # the strength value
    R = np.zeros(npop, dtype=int)  # the raw fitness
    dom = np.full((npop, npop), False)  # domination matrix
    for i in range(npop - 1):
        for j in range(i, npop):
            if dominates(objs[i], objs[j]):
                S[i] += 1
                dom[i, j] = True
            elif dominates(objs[j], objs[i]):
                S[j] += 1
                dom[j, i] = True
    for i in range(npop):
        R[i] = np.sum(S[dom[:, i]])
    dis = np.full((npop, npop), np.inf)
    for i in range(npop):
        temp_objs = np.max((objs, np.tile(objs[i], (npop, 1))), axis=0)
        for j in range(npop):
            if i != j:
                dis[i, j] = np.sqrt(np.sum((objs[i] - temp_objs[j]) ** 2))
    D = 1 / (np.sort(dis)[:, K] + 2)  # density
    F = R + D  # fitness
    return F


def truncation(objs, num):
    # truncate part of population
    npop = objs.shape[0]
    dis = np.full((npop, npop), np.inf)
    for i in range(npop):
        temp_objs = np.max((objs, np.tile(objs[i], (npop, 1))), axis=0)
        for j in range(npop):
            if i != j:
                dis[i, j] = np.sqrt(np.sum((objs[i] - temp_objs[j]) ** 2))
    delete = np.full(npop, False)
    while np.sum(delete) < num:
        remain = np.where(~delete)[0]
        temp = np.sort(dis[remain][:, remain])
        delete[remain[np.argmin(temp[:, 0])]] = True
    return delete


def environmental_selection(pop, objs, npop):
    # environmental selection
    fitness = cal_fitness(objs)
    index = fitness < 1
    if np.sum(index) <= npop:
        rank = np.argsort(fitness)
        return pop[rank[: npop]], objs[rank[: npop]], fitness[rank[: npop]]
    delete = truncation(objs[index], np.sum(index) - npop)
    index[np.where(index)[0][delete]] = False
    return pop[index], objs[index], fitness[index]


def main(npop, iter, lb, ub, nobj=3, eta_c=20, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space (default = 3)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # the objectives of population
    fitness = cal_fitness(objs)  # fitness

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 50 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Crossover + mutation
        mating_pool = selection(pop, fitness, 1)
        off = crossover(mating_pool, lb, ub, 1, eta_c)
        off = mutation(off, lb, ub, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Environmental selection
        pop, objs, fitness = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), npop)

    # Step 3. Sort the results
    pf = objs[fitness < 1]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ1')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 400, np.array([0] * 7), np.array([1] * 7))
