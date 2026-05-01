import pandas as pd
import numpy as np
import os
from typing import Optional

from Surrogate import RandomForestSurrogate
from Acquisition import ExpectedImprovement


def bayesian_search(
    file_path: str,
    budget: int,
    output_file: str,
    n_initial: int = 10,
    n_candidates: int = 500,
    random_state: Optional[int] = None,
):
    rng = np.random.default_rng(random_state)

    data = pd.read_csv(file_path)
    config_columns = data.columns[:-1].tolist()
    performance_column = data.columns[-1]

    system_name = os.path.basename(file_path).split(".")[0].lower()
    maximization = system_name in {"---"}

    if maximization:
        worst_value = data[performance_column].min() / 2
    else:
        worst_value = data[performance_column].max() * 2

    def lookup(config):
        mask = (data[config_columns] == pd.Series(config, index=config_columns)).all(axis=1)
        row = data.loc[mask]
        return float(row[performance_column].iloc[0]) if not row.empty else worst_value

    def sample_random_configs(n):
        return np.column_stack([rng.choice(data[col].unique(), size=n) for col in config_columns])

    def is_better(a, b):
        return a > b if maximization else a < b

    surrogate = RandomForestSurrogate(random_state=None if random_state is None else int(random_state))
    ei = ExpectedImprovement(ee=0.05)

    observed_configurations = []
    observed_performance = []
    search_results = []
    convergence_curve = []

    best_performance = -np.inf if maximization else np.inf
    best_solution = []




    # warm-up with latin hypercube sampling
    lhs_configs = np.column_stack([
        rng.permutation(np.resize(data[col].unique(), n_initial))
        for col in config_columns
    ])

    

    for config in lhs_configs:
        config = [int(v) for v in config]
        performance = lookup(config)
        observed_configurations.append(config)
        observed_performance.append(performance)

    for iteration in range(budget):


        X_obs = np.array(observed_configurations)
        y_obs = np.array(observed_performance)
        y_fit = -y_obs if maximization else y_obs

        surrogate.fit(X_obs, y_fit)

        candidates = sample_random_configs(n_candidates)
        mu, sigma = surrogate.guess(candidates)

        best_so_far = -best_performance if maximization else best_performance
        ei_scores = ei.evaluate(mu, sigma, best_so_far)

        best_idx = int(np.argmax(ei_scores))
        config = [int(v) for v in candidates[best_idx]]

        performance = lookup(config)
        observed_configurations.append(config)
        observed_performance.append(performance)

        if is_better(performance, best_performance):
            best_performance = performance
            best_solution = config

        search_results.append(config + [performance])
        convergence_curve.append(best_performance)

    columns = config_columns + ["Performance"]
    pd.DataFrame(search_results, columns=columns).to_csv(output_file, index=False)

    return [int(x) for x in best_solution], best_performance, convergence_curve


def main():
    datasets_folder = "datasets"
    output_folder = "search_results"
    os.makedirs(output_folder, exist_ok=True)
    budget = 100

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_bo_results.csv")
            best_solution, best_performance, _ = bayesian_search(file_path, budget, output_file, random_state=67)
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance
            }

    for system, result in results.items():
        print(f"System: {system}")
        print(f"  Best Solution:    [{', '.join(map(str, result['Best Solution']))}]")
        print(f"  Best Performance: {result['Best Performance']}")

if __name__ == "__main__":
    main()