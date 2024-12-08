import numpy as np
import random
import statistics

from NMOD_lab2.conditional_print import ConditionalPrint

max_nn_algorithm_iteration = 10

y1 = np.array([1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]) # 3
y2 = np.array([1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1]) # 8
y3 = np.array([1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]) # 4
y4 = np.array([0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]) # 9

Y = np.array([y1,y2,y3,y4])

def make_bits_noisy(y, bits_count):
    noisy_positions = random.sample(range(len(y)), bits_count)
    y_noisy = y.copy()
    for i in noisy_positions:
        y_noisy[i] = y_noisy[i] ^ 1
    return y_noisy

def sign(x):
    if isinstance(x, (list, np.ndarray)):
        return np.array([1 if el > 0 else 0 for el in x])
    elif isinstance(x, (int, float)):
        return 1 if x > 0 else 0

def async_method(y_noisy, W, y_original, enable_print=False):
    y_noisy = y_noisy.copy()
    with ConditionalPrint(enable_print):
        print(f"y_original = {y_original}")
        print(f"y_noizy    = {y_noisy}")
        for i in range(max_nn_algorithm_iteration):
            y_in = y_noisy.copy()
            print(f"\nStage {i+1}:")
            for j in range(len(y_noisy)):
                s_j = np.dot(y_noisy, W[:, j])
                y_noisy[j] = sign(s_j)
                print(f"y_model({j+1:02}) = [{" ".join(f" {x} " if i != j else f"({x})" for i, x in enumerate(y_noisy))}]")
            if np.array_equal(y_in, y_noisy):
                if np.array_equal(y_in, y_original):
                    print(f"y_stage_{i+1} == y_original, relaxation with correct value")
                    return True
                else:
                    print(f"y_stage_{i+1} == y_stage_{i} != y_original, relaxation with wrong value")
                    return False
            else:
                print(f"y_stage{i+1} != y_stage{i}, continue calculation")
        print(f"model can’t find relaxation, max iteration = {max_nn_algorithm_iteration}")
        return False

def sync_method(y_noisy, W, y_original, enable_print=False):
    with ConditionalPrint(enable_print):
        print(f"y_original = {y_original}")
        print(f"y_noizy    = {y_noisy}")
        y_out = y_noisy.copy()
        for i in range(max_nn_algorithm_iteration):
            print(f"\nStage {i+1}:")
            y_in = y_out
            s = np.dot(y_in, W)
            y_out = sign(s)
            print(f"y_model({i+1}) = {y_out}")
            if np.array_equal(y_out, y_in):
                if np.array_equal(y_out, y_original):
                    print(f"y_stage_{i + 1} == y_original, relaxation with correct value")
                    return True
                else:
                    print(f"y_stage_{i + 1} == y_stage_{i} != y_original, relaxation with wrong value")
                    return False
            else:
                print(f"y_stage{i + 1} != y_stage_{i}, continue calculation")
        print(f"model can’t find relaxation, max iteration = {max_nn_algorithm_iteration}")
        return False

def main():
    print("\nSource vectors:")
    for y_idx, y_original in enumerate(Y):
        print(f"y{y_idx+1} = {y_original}")

    W = ((np.dot(np.matrix.transpose(np.multiply(Y, 2) - 1), (np.multiply(Y, 2)) - 1)) -
         np.identity(y1.size))

    for y_idx, y_original in enumerate(Y):
        y_noisy = make_bits_noisy(y_original, 1)
        print(f"\n\tAsync y{y_idx+1}:")
        async_method(y_noisy, W, y_original, enable_print=True)
        print(f"\n\tSync y{y_idx+1}:")
        sync_method(y_noisy, W, y_original, enable_print=True)


    measurements_number = 11
    measurements_output_async = {f"y_{i+1}": [] for _ in range(measurements_number) for i, y_i in enumerate(Y)}
    measurements_output_sync = {f"y_{i+1}": [] for _ in range(measurements_number) for i, y_i in enumerate(Y)}

    for measurement_idx in range(measurements_number):
        for y_idx, y_original in enumerate(Y):
            max_recognized_bits_async = 0
            max_recognized_bits_sync = 0
            for noisy_bits_count in range(1, len(Y[y_idx])+1):
                y_noisy = make_bits_noisy(y_original, noisy_bits_count)
                if async_method(y_noisy, W, y_original):
                    max_recognized_bits_async = noisy_bits_count
                if sync_method(y_noisy, W, y_original):
                    max_recognized_bits_sync = noisy_bits_count
            measurements_output_async[f"y_{y_idx+1}"].append(max_recognized_bits_async)
            measurements_output_sync[f"y_{y_idx+1}"].append(max_recognized_bits_sync)

    measurements_output_async = {key: statistics.median(values_list)
                                 for (key, values_list) in measurements_output_async.items()}
    measurements_output_sync = {key: statistics.median(values_list)
                                for (key, values_list) in measurements_output_sync.items()}

    print(f"\nMedian number of recognised noisy bits ({measurements_number} samples):")
    print("Async:", measurements_output_async)
    print("Sync: ", measurements_output_sync)

    # for i, y_i in enumerate(Y):
    #     print("\nAsync:")
    #     for j in range(1, len(Y[i])+1):
    #         print(f"y_{i+1} with {j:02} noisy bit{" " if j == 1 else "s"} -> ",
    #               async_method(make_bits_noisy(y_i, j), W, y_i))
    #     print("\nSync:")
    #     for j in range(1, len(Y[i]) + 1):
    #         print(f"y_{i + 1} with {j:02} noisy bit{" " if j == 1 else "s"} -> ",
    #               sync_method(make_bits_noisy(y_i, j), W, y_i))

if __name__ == '__main__':
    main()