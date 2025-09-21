
"""A simple Genetic Algorithm to optimize daily usage fraction to minimize overflow.
This is a demonstrative GA — replace or expand for production use.
"""
import argparse
import pandas as pd
import numpy as np
import json
import os
import random

def load_data(path):
    df = pd.read_csv(path, parse_dates=True)
    # attempt to find rainfall column
    rain_cols = [c for c in df.columns if 'rain' in c.lower() or 'precip' in c.lower() or 'mm' in c.lower()]
    if rain_cols:
        rain = df[rain_cols[0]].fillna(0).astype(float).values
    else:
        # fallback numeric
        rain = df.select_dtypes('number').iloc[:,0].fillna(0).astype(float).values
    return rain

def simulate(schedule, rain, tank_capacity=10000, catchment_coeff=0.8, area_m2=50, init_storage=2000):
    """Simulate tank level over days.
    schedule: fraction [0..1] to release/use each day (array same length as rain)
    rain: daily rainfall in mm array
    tank capacity in liters. area_m2 * rain(mm) * 1 liter/mm/m2 = liters input.
    """
    storage = init_storage
    overflow = 0.0
    used = 0.0
    for i, r in enumerate(rain):
        inflow = area_m2 * r * catchment_coeff  # liters
        storage += inflow
        if storage > tank_capacity:
            overflow += storage - tank_capacity
            storage = tank_capacity
        # use according to schedule fraction but not exceeding storage
        use = min(storage, schedule[i]*tank_capacity)
        storage -= use
        used += use
    # objective: maximize used and minimize overflow -> combine into fitness
    # we return higher fitness for better solutions.
    fitness = used - overflow*2.0
    return fitness, {'used': used, 'overflow': overflow, 'final_storage': storage}

def random_schedule(length):
    return np.random.rand(length)

def crossover(a, b):
    # single-point crossover
    p = random.randint(1, len(a)-1)
    child = np.concatenate([a[:p], b[p:]])
    return child

def mutate(a, rate=0.05):
    for i in range(len(a)):
        if random.random() < rate:
            a[i] = random.random()
    return a

def ga_optimize(rain, generations=50, pop_size=30, tank_capacity=10000):
    length = len(rain)
    pop = [random_schedule(length) for _ in range(pop_size)]
    best = None
    for g in range(generations):
        scored = []
        for ind in pop:
            fit, info = simulate(ind, rain, tank_capacity=tank_capacity)
            scored.append((fit, ind, info))
        scored.sort(key=lambda x: x[0], reverse=True)
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
        # selection top 50%
        selected = [ind for (_, ind, _) in scored[:pop_size//2]]
        # create children
        children = []
        while len(children) < pop_size - len(selected):
            a = random.choice(selected)
            b = random.choice(selected)
            child = crossover(a, b)
            child = mutate(child)
            children.append(child)
        pop = selected + children
        if g % 10 == 0:
            print(f'Gen {g} best fitness {scored[0][0]:.2f} used {scored[0][2]["used"]:.1f} overflow {scored[0][2]["overflow"]:.1f}')
    return best

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--pop', type=int, default=30)
    parser.add_argument('--tank', type=float, default=10000.0)
    parser.add_argument('--out', default='models/ga_best.json')
    args = parser.parse_args()
    # Resolve relative data path against script directory
    if not os.path.isabs(args.data):
        args.data = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data)


    rain = load_data(args.data)
    best = ga_optimize(rain, generations=args.generations, pop_size=args.pop, tank_capacity=args.tank)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result = {
        'fitness': float(best[0]),
        'info': best[2]
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print("GA complete. Best fitness:", best[0])
