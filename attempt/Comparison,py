import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind

from RandomSearch import random_search
from Optimizer import bayesian_search


def main():
    datasets_folder = "datasets"
    output_folder = "comparison_results"
    os.makedirs(output_folder, exist_ok=True)

    budget = 100
    n_initial =40
    n_trials = 20

    for file_name in os.listdir(datasets_folder):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(datasets_folder, file_name)
        system = file_name.split(".")[0]

        rs_curves, bo_curves = [], []

        for trial in range(n_trials):
            _, _, rs_curve = random_search(file_path, budget, output_file=os.devnull, random_state=trial)
            _, _, bo_curve = bayesian_search(file_path, budget, output_file=os.devnull, n_initial=n_initial, random_state=trial)
            rs_curves.append(rs_curve)
            bo_curves.append(bo_curve)

        rs_arr = np.array(rs_curves)   # (n_trials, budget)
        bo_arr = np.array(bo_curves)
        rs_final = rs_arr[:, -1]
        bo_final = bo_arr[:, -1]

        t_stat, p_value = ttest_ind(bo_final, rs_final)

        # ── Convergence plot ───────────────────────────────────────────────
        iters = np.arange(1, budget + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        for arr, color, label in [(rs_arr, "tab:orange", "Random search"), (bo_arr, "tab:blue", "Bayesian opt")]:
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            ax.plot(iters, mean, color=color, label=label)
            ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=color)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best performance")
        ax.set_title(f"{system} — convergence ({n_trials} trials, mean ± 1 std)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"{system}_convergence.png"), dpi=150)
        plt.close(fig)

        # ── Box plot ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.boxplot([rs_final, bo_final], labels=["Random search", "Bayesian opt"],
                   medianprops=dict(color="black"))
        ax.set_ylabel("Final performance")
        ax.set_title(f"{system} — final performance")
        ax.set_xlabel(f"t={t_stat:.3f},  p={p_value:.4f}{'  *' if p_value < 0.05 else ''}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"{system}_boxplot.png"), dpi=150)
        plt.close(fig)

        # ── T-test summary ─────────────────────────────────────────────────
        print(f"System: {system}")
        print(f"  RS  final: {rs_final.mean():.4f} ± {rs_final.std():.4f}")
        print(f"  BO  final: {bo_final.mean():.4f} ± {bo_final.std():.4f}")
        print(f"  t={t_stat:.3f}  p={p_value:.4f}  {'(significant)' if p_value < 0.05 else '(not significant)'}")


if __name__ == "__main__":
    main()