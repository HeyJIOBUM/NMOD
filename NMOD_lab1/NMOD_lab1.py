import math

import numpy as np
import random
import matplotlib.pyplot as plt

# y = a*sin(b*x)+d
a = 1
b = 9
d = 0.5
input_neurons = 4
period = 2 * math.pi / b

learning_rate = 0.35
epsilon = 1e-10

dataset_size = 45
train_dataset_size = 30
test_dataset_size = dataset_size - train_dataset_size

weights = np.array([random.random() - 0.5 for _ in range(input_neurons)])
threshold = random.random()

def function(x):
    return np.multiply(a, np.sin(np.multiply(b, x))) + d

def main():
    global weights, threshold

    print("\n"
          "initial values: \n",
          "\tweights: " + ", ".join(map(str, weights)),"\n",
          "\tthreshold: " + str(threshold))

    print("\n"
          "Stage 1: data preparing"
          "\n")

    x_input_data = [x * period / (train_dataset_size - 1) for x in range(dataset_size)]
    for i, x in enumerate(x_input_data):
        print(f"x{i+1} = {x}; y{i} = {function(x)}")

    print("\n"
          "Stage 2: split data on train/test data")

    print("\n"
          "train_data:")
    x_train_data = x_input_data[:train_dataset_size]
    for i, x in enumerate(x_train_data):
        print(f"x{i+1} = {x}; y{i+1} = {function(x)}")

    print("\n"
          "test_data:")
    x_test_data = x_input_data[train_dataset_size:]
    for i, x in enumerate(x_test_data):
        idx = i + train_dataset_size + 1
        print(f"x{idx} = {x}; y{idx} = {function(x)}")

    print("\n"
          "Stage 3: prepare train/test data for NN")

    print("\n"
          "train_data:")
    prepared_train_data = [function(x) for x in x_train_data]
    for i in range(len(prepared_train_data) - input_neurons):
        print(", ".join(f'{el:.4f}' for el in prepared_train_data[i:i + input_neurons]) +
              f" -> {prepared_train_data[i+input_neurons]:.4f}")

    print("\n"
          "test_data:")
    correct_test_output = list(map(function, x_test_data))
    prepared_test_data = prepared_train_data[-input_neurons:] + correct_test_output
    print(", ".join(f"{el:.4f}" for el in prepared_train_data[-input_neurons:]) + f" -> y'{train_dataset_size+1}")

    print("\n"
          "Stage 4: train & test model")

    epoch = 1
    while epoch <= 50:
        print(f'Epoch #{epoch}')

        train_error = 0
        for i in range(len(prepared_train_data) - input_neurons):
            inputs = prepared_train_data[i:i + input_neurons]
            e = prepared_train_data[i + input_neurons]
            y = np.dot(inputs, weights) - threshold
            weights = weights - [learning_rate * (y - e) * x for x in inputs]
            threshold = threshold + learning_rate * (y - e)
            train_error += (y - e)**2 / 2
        avg_train_error = train_error / (len(prepared_train_data) - input_neurons)
        print(f"train error: {avg_train_error}\t")

        test_error = 0
        for i, e in enumerate(correct_test_output):
            inputs = prepared_test_data[i:i + input_neurons]
            y = np.dot(inputs, weights) - threshold
            prepared_test_data[i + input_neurons] = y
            test_error += (y - e)**2 / 2
        avg_test_error = test_error / len(correct_test_output)
        print(f"test error: {avg_test_error}")

        if avg_test_error > epsilon:
            print(f"test error: {avg_test_error} > {epsilon}, next epoch")
            epoch += 1
        else:
            print(f"test error: {avg_test_error} <= {epsilon}, stop learning")
            break

    print("\n"
          "Stage 5: print full model outputs for best epoch")

    print(f"Last epoch: {epoch}")
    for i, e in enumerate(correct_test_output):
        inputs = prepared_test_data[i:i + input_neurons]
        y = prepared_test_data[i + input_neurons]
        print(", ".join(f"{el:.4f}" for el in inputs) + f" -> {y:.6f}({e:.6f})")

    print("\n"
          "end values: \n",
          "\tweights: " + ", ".join(map(str, weights)),"\n",
          "\tthreshold: " + str(threshold))

    plt.figure()
    plt.subplot(211)
    plt.plot(x_input_data, function(x_input_data), 'b')

    nn_output = [function(x_input_data[i]) for i in range(input_neurons)]
    for i in range(input_neurons, len(x_input_data)):
        nn_output.append(np.dot(nn_output[-input_neurons:], weights) - threshold)

    plt.subplot(212)
    plt.plot(x_input_data, nn_output, 'r')
    plt.show()


if __name__ == "__main__":
    main()

