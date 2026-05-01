import pandas as pd
import numpy as np
import os


def random_search(file_path, budget, output_file, random_state=None):
    rng = np.random.default_rng(random_state)
    data = pd.read_csv(file_path)

    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]

    system_name = os.path.basename(file_path).split('.')[0]
    if system_name.lower() == "---":
        maximization = True
    else:
        maximization = False

    if maximization:
        worst_value = data[performance_column].min() / 2
    else:
        worst_value = data[performance_column].max() * 2

    best_performance = -np.inf if maximization else np.inf
    best_solution = []
    search_results = []
    convergence_curve = []

    for _ in range(budget):
        sampled_config = [int(rng.choice(data[col].unique())) for col in config_columns]

        matched_row = data.loc[(data[config_columns] == pd.Series(sampled_config, index=config_columns)).all(axis=1)]

        if not matched_row.empty:
            performance = matched_row[performance_column].iloc[0]
        else:
            performance = worst_value

        if maximization:
            if performance > best_performance:
                best_performance = performance
                best_solution = sampled_config
        else:
            if performance < best_performance:
                best_performance = performance
                best_solution = sampled_config

        search_results.append(sampled_config + [performance])
        convergence_curve.append(best_performance)

    columns = list(config_columns) + ["Performance"]
    search_df = pd.DataFrame(search_results, columns=columns)
    search_df.to_csv(output_file, index=False)

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
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
            best_solution, best_performance, _ = _, _, rs_curve = random_search(file_path, budget, output_file, random_state=1)
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