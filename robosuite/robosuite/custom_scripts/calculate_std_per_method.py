import numpy as np

# My experimental data tables: Each key is a method, and values are results across tasks.
t1 = {
    "Method_A": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Method_B": [0.1, 1.0, 0.8, 0.2, 0.2, 0.0, 0.0, 0.0]
}

t2 = {
    "Method_A": [0.1, 1.0, 0.8, 0.2, 0.2, 0.0, 0.0, 0.0],
    "Method_B": [0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Method_C": [0.1, 1.0, 0.9, 0.1, 0.4, 0.0, 0.0, 0.0],
    "Method_D": [0.0, 1.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0]
}

t3 = {
    "Method_A": [0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 0.6, 0.7],
    "Method_B": [0.5, 0.8, 1.0, 0.3, 0.2, 0.3, 0.3, 0.7],
    "Method_C": [0.4, 0.8, 0.9, 0.3, 0.2, 0.4, 0.3, 0.2],
    "Method_D": [0.4, 0.8, 0.9, 0.4, 0.3, 0.2, 0.3, 0.3]
}

tables = [t1, t2, t3]

for i, data in enumerate(tables):
    print(f"*** Table {i} ***")
    # Calculate mean and standard deviation for each method
    for method, scores in data.items():
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)  # ddof=1 for sample SD (n-1 denominator)
        print(f"{method}: Mean = {mean:.2f}, SD = {std:.2f}")
