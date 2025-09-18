import numpy as np


def binomial_std(p, n):
    return np.sqrt(p * (1 - p)) / np.sqrt(n)  # Equivalent to sqrt(p(1-p)/n)


# My experimental data tables: Each key is a method, and values are results across tasks.
t1 = {
    "Method_A": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Method_B": [0.1, 1.0, 0.8, 0.2, 0.2, 0.0, 0.0, 0.0]
}

t2 = {
    "Method_A": [0.1, 1.0, 0.8, 0.2, 0.2, 0.0, 0.0, 0.0],
    "Method_B": [0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Method_C": [0.1, 1.0, 0.9, 0.1, 0.4, 0.0, 0.0, 0.0],
    "Method_D": [0.0, 1.0, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0],
    "Method_E": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Method_F": [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
}

t3 = {
    "Method_A": [0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 0.6, 0.7],
    "Method_B": [0.5, 0.8, 1.0, 0.3, 0.2, 0.3, 0.3, 0.7],
    "Method_C": [0.4, 0.8, 0.9, 0.3, 0.2, 0.4, 0.3, 0.2],
    "Method_D": [0.4, 0.8, 0.9, 0.4, 0.3, 0.2, 0.3, 0.3],
    "Method_E": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Method_F": [0.4, 0.7, 0.6, 0.3, 0.1, 0.4, 0.4, 0.0]
}

tables = [t1, t2, t3]
n = 10    # Number of trials

for i, data in enumerate(tables):
    print(f"*** Table {i} ***")
    # Calculate standard deviation for each rate
    for method, scores in data.items():
        p_array = np.array(scores)
        std_array = binomial_std(p_array, n)
        latex_pairs = [f"{p:.1f}(±{std:.2f})" for p, std in zip(p_array, std_array)]
        print(f"{method}:")
        print(f"SD = {np.round(std_array, 2)}")
        print(f"LaTeX = {' & '.join(latex_pairs)}")
