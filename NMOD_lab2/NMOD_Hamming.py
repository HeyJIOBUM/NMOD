import statistics

import numpy as np
import random

from NMOD_lab2.conditional_print import ConditionalPrint

max_iteration = 40

y1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])  # 3
y2 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])  # 8
y3 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])  # 4
y4 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])  # 9

Y = np.array([y1,y2,y3,y4])


W = np.multiply(Y,0.5)
n = len(Y[0])
T = n / 2
e = 1 / n


def ReLU(s):
    return np.array([max(0, x) for x in s])

def make_bits_noisy(y, bits_count):
    noisy_positions = random.sample(range(len(y)), bits_count)
    y_noisy = y.copy()
    for i in noisy_positions:
        y_noisy[i] ^= 1
    return y_noisy

def calculate_winner(y_noizy, y_original, enable_print=False):
    with ConditionalPrint(enable_print):
        print(f"y_original = {y_original}")
        print(f"y_noizy    = {y_noizy}\n")
        y = np.array([np.dot(W[i], y_noizy) + T for i in range(len(Y))])
        z = y
        for i in range(max_iteration):
            s = [sum([z[k] if i == k else -e * z[k] for k in range(len(Y))]) for i in range(len(Y))]
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            z = ReLU(s)
            print(f"winner({i:02}) = {z}")
            pos_nums = {i: el
                        for i, el in enumerate(z) if el > 0}
            if len(pos_nums) == 1:
                y_model = Y[next(iter(pos_nums))]
                print(f"y_model({i}) = {y_model}")
                if np.array_equal(y_original, y_model):
                    print(f"y_model({i}) == y_original, correct")
                    return True
                else:
                    print(f"y_model({i}) != y_original, wrong")
                    return False
            elif len(pos_nums) == 0:
                print(f"model can’t find relaxation")
                return False
        else:
            print(f"model can’t find relaxation, max iteration = {max_iteration}")
            return False

def main():
    print("\nSource vectors:")
    for i, y_i in enumerate(Y):
        print(f"y{i+1} = {y_i}")

    for y_idx, y_original in enumerate(Y):
        print(f"\n\tWinner y{y_idx+1}:")
        calculate_winner(make_bits_noisy(y_original, 2), y_original, enable_print=True)

    measurements_number = 10
    measurements_output = {f"y_{i+1}": [] for _ in range(measurements_number) for i, y_i in enumerate(Y)}

    for measurement_idx in range(measurements_number):
        for y_idx, y_original in enumerate(Y):
            max_recognized_bits = 0
            for noisy_bits_count in range(1, len(Y[y_idx]) + 1):
                y_noisy = make_bits_noisy(y_original, noisy_bits_count)
                if calculate_winner(y_noisy, y_original):
                    max_recognized_bits = noisy_bits_count
            measurements_output[f"y_{y_idx+1}"].append(max_recognized_bits)

    measurements_output = {key: statistics.median(values_list)
                                for (key, values_list) in measurements_output.items()}

    print(f"\nMedian number of recognised noisy bits ({measurements_number} samples):\n", measurements_output)


if __name__ == '__main__':
    main()


