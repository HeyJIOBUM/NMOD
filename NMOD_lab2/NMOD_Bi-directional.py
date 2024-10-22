import statistics

import numpy as np
import random

from NMOD_lab2.conditional_print import ConditionalPrint

max_iteration = 10

x1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0])  # 3
x2 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1])  # 8
x3 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])  # 4
x4 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])  # 9

y1 = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])  # 3
y2 = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])  # 8
y3 = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])  # 4
y4 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])  # 9

X = np.array([x1, x2, x3, x4])
X[X==0] = -1

Y = np.array([y1, y2, y3, y4])
Y[Y==0] = -1

def make_bits_noisy(y, bits_count):
    noisy_positions = random.sample(range(len(y)), bits_count)
    y_noisy = y.copy()
    for i in noisy_positions:
        y_noisy[i] = y_noisy[i] * -1
    return y_noisy

def calculate_x(y_noisy, W, y_original, enable_print=False):
    with ConditionalPrint(enable_print):
        print(f"y_original = {y_original}")
        print(f"y_model(0) = {y_noisy}")
        y_last = y_noisy
        for i in range(max_iteration):
            print(f"\nStage {i + 1}:")
            s_x = np.dot(y_last, np.matrix.transpose(W))
            x_out = np.sign(s_x)
            print(f"x_model({i + 1}) = {x_out}")
            s_y = np.dot(x_out, W)
            y_out = np.sign(s_y)
            print(f"y_model({i + 1}) = {y_out}")
            if np.array_equal(y_last, y_out):
                print(f"y_model({i + 1}) == y_model({i}), relaxation")
                if np.array_equal(y_last, y_original):
                    print(f"y_model({i + 1}) == y_original, relaxation with correct value")
                    return True
                else:
                    print(f"y_model({i + 1}) != y_original, relaxation with wrong value")
                    return False
            else:
                y_last = y_out
                print(f"y_model({i + 1}) != y_model({i}), continue calculation")
        print(f"model can’t find relaxation, max iteration = {max_iteration}")
        return False

def calculate_y(x_noisy, W, x_original, enable_print=False):
    with ConditionalPrint(enable_print):
        print(f"x_original = {x_original}")
        print(f"x_model(0) = {x_noisy}")
        x_last = x_noisy
        for i in range(max_iteration):
            print(f"\nStage {i + 1}:")
            s_y = np.dot(x_last, W)
            y_out = np.sign(s_y)
            print(f"y_model({i + 1}) = {y_out}")
            s_x = np.dot(y_out, np.matrix.transpose(W))
            x_out = np.sign(s_x)
            print(f"x_model({i + 1}) = {x_out}")
            if np.array_equal(x_last, x_out):
                print(f"x_model({i + 1}) == x_model({i}), relaxation")
                if np.array_equal(x_last, x_original):
                    print(f"x_model({i + 1}) == x_original, relaxation with correct value")
                    return True
                else:
                    print(f"x_model({i + 1}) != x_original, relaxation with wrong value")
                    return False
            else:
                x_last = x_out
                print(f"x_stage({i + 1}) != x_stage({i}), continue calculation")
        print(f"model can’t find relaxation, max iteration = {max_iteration}")
        return False

def main():
    W = np.dot(np.matrix.transpose(X), Y)

    print("\nSource vectors:")
    for i in range(len(X)):
        print(f"x{i+1} = {X[i]}; y{i+1} = {Y[i]}")

    for i, y_i in enumerate(Y):
        print(f"\n\ty{i+1}:")
        calculate_x(make_bits_noisy(y_i, 2), W, y_i, enable_print=True)

    for i, x_i in enumerate(X):
        print(f"\n\tx{i+1}:")
        calculate_y(make_bits_noisy(x_i, 2), W, x_i, enable_print=True)

    measurements_number = 10
    measurements_output_y = {f"y_{i}": [] for _ in range(measurements_number) for i, y_i in enumerate(Y)}
    measurements_output_x = {f"x_{i}": [] for _ in range(measurements_number) for i, x_i in enumerate(Y)}

    for measurement_idx in range(measurements_number):
        for y_idx, y_original in enumerate(Y):
            max_recognized_bits = 0
            for noisy_bits_count in range(1, len(Y[y_idx])+1):
                y_noisy = make_bits_noisy(y_original, noisy_bits_count)
                if calculate_x(y_noisy, W, y_original):
                    max_recognized_bits = noisy_bits_count
            measurements_output_y[f"y_{y_idx}"].append(max_recognized_bits)

        for x_idx, x_original in enumerate(X):
            max_recognized_bits = 0
            for noisy_bits_count in range(1, len(X[x_idx]) + 1):
                x_noisy = make_bits_noisy(x_original, noisy_bits_count)
                if calculate_y(x_noisy, W, x_original):
                    max_recognized_bits = noisy_bits_count
            measurements_output_x[f"x_{x_idx}"].append(max_recognized_bits)

    measurements_output_y = {key: statistics.median(values_list)
                                 for (key, values_list) in measurements_output_y.items()}
    measurements_output_x = {key: statistics.median(values_list)
                                for (key, values_list) in measurements_output_x.items()}

    print(f"\nMedian number of recognised noisy bits ({measurements_number} samples):")
    print("Y:", measurements_output_y)
    print("X:", measurements_output_x)

    # print("\nBi-directional find X:")
    # for i, y_i in enumerate(Y):
    #     print(f"\ny_{i + 1}:")
    #     for j in range(1, len(Y[i])+1):
    #         print(f"y_{i+1} with {j:02} noisy bit{" " if j == 1 else "s"} -> ",
    #               calculate_x(make_bits_noisy(y_i, j), W, y_i))
    #
    # print("\nBi-directional find Y:")
    # for i, x_i in enumerate(X):
    #     print(f"\nx_{i+1}:")
    #     for j in range(1, len(Y[i])+1):
    #         print(f"x_{i+1} with {j:02} noisy bit{" " if j == 1 else "s"} -> ",
    #               calculate_y(make_bits_noisy(x_i, j), W, x_i))

    pass

if __name__ == '__main__':
    main()
