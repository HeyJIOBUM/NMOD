import random
import math
import numpy as np
import torch.nn

a = 4
b = 7
d = 0.2
ins = 4


def func(x):
    return a * np.sin(b*x) + d


def nn_func(args, weights, shift):
    return np.dot(args, weights) - shift


def prepare(start, step):
    period = 2 * math.pi / b
    x = [start + i * period/30 for i in range(45)]
    y = [func(x) for x in x]
    return x, y


def stage1(x, y):
    print("\n\tStage 1: data preparing")
    for i in range(1, 46):
        print(f'x{i} = {x[i-1]};\ty{i} = {y[i-1]}')


def stage2(x, y):
    print("\n\tStage 2: split data on train/test data")

    print("train_data:")
    train_data = x[0:30]
    for i in range(1, 31):
        print(f'x{i} = {x[i-1]};\ty{i} = {y[i-1]}')

    print("test_data:")
    test_data = x[30:45]
    for i in range(31, 46):
        print(f'x{i} = {x[i-1]};\ty{i} = {y[i-1]}')
    return train_data, test_data


def stage3(x, y, ins_count):
    print("\n\tStage 3: prepare train/test data for NN")
    print("train_data:")
    for i in range(30-ins_count):
        print(" ".join(f'{val:.4f}' for val in y[i:i + ins_count]), f"-> {y[i+ins_count]:.4f}")
    print("test_data")
    print(" ".join(f'{val:.4f}' for val in y[30-ins_count:30]), "-> ?")


def stage4(x, y, a, err, sep):
    print("\n\tStage 4: train & test model:")
    iteration = 1
    weights = np.array([random.random()-.5 for _ in range(ins)])
    shift = random.random()
    while iteration < 1000:
        print(f"Epoch #{iteration}")

        train_loss = 0
        for i in range(sep-ins):
            train_loss += (y[i+ins] - nn_func(y[i:i+ins], weights, shift))**2

        test_loss = 0
        for i in range(sep-ins, len(x)-ins):
            test_loss += (y[i+ins] - nn_func(y[i:i+ins], weights, shift))**2
        print(f"train_loss: {train_loss} \t test_loss: {test_loss}")
        if test_loss > err:
            print(f"test_loss > {err} -> continue training")
        else:
            print(f"test_loss <= {err} -> stop  training")
            break

        for i in range(sep-ins):
            # for j in range(ins):
            #     weights[j] = weights[j] - a*(nn_func(y[i:i+ins], weights, shift) - y[i+ins])*y[i+j]
            weights = weights - [a*(nn_func(y[i:i+ins], weights, shift) - y[i+ins])*y[i+j] for j in range(ins)]
            shift = shift + a*(nn_func(y[i:i+ins], weights, shift) - y[i+ins])

        iteration += 1
    return weights, shift


def stage5(x, y, weights, shift, ins_count):
    print("\n\tStage 5: print full model outputs for best epoch")
    for i in range(30-ins_count, len(x)-ins_count):
        print(" ".join(f'{val:.4f}' for val in y[i:i + ins_count]),
              f"-> {nn_func(y[i:i+ins_count], weights, shift):.4f}"
              f"({y[i+ins_count]:.4f})")


print("\tFunction: y = a sin sin (bx) + d")
x, y = prepare(0, 0.5)

stage1(x, y)
train, test = stage2(x, y)
stage3(x, y, ins)
weights, shift = stage4(x, y, 0.026, 0.000001, 30)
stage5(x, y, weights, shift, ins)
