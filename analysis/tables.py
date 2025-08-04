import numpy as np
import os
from scipy.stats import sem, t
from configs.config_global import FIG_DIR

def summarize_model_performance_latex(data, model_names, dataset_names, precision=3, save_dir=None, table_name=None, best='max'):
    n_models = len(model_names)
    n_datasets = len(dataset_names)

    # Compute mean and 95% CI
    means = np.zeros((n_models, n_datasets))
    cis = np.zeros((n_models, n_datasets))

    for i in range(n_models):
        for j in range(n_datasets):
            values = data[i][j]
            values = np.array(values)
            n = len(values)
            means[i, j] = np.mean(values)

            if n > 1:
                std_err = sem(values)
                cis[i, j] = std_err * t.ppf(0.975, df=n - 1)
            else:
                raise ValueError(f"Not enough data points for model {model_names[i]} on dataset {dataset_names[j]}.")

    # Escape underscores in LaTeX names
    model_names = [name.replace('_', '\\_') for name in model_names]
    dataset_names = [name.replace('_', '\\_') for name in dataset_names]

    # Build rows
    latex_rows = []
    for i, model in enumerate(model_names):
        row = [model]
        for j in range(n_datasets):
            mean = means[i, j]
            ci = cis[i, j]
            entry = f"{mean:.{precision}f} \\scriptsize{{±{ci:.{precision}f}}}"
            row.append(entry)
        latex_rows.append(row)

    # Bold best model per column
    for j in range(n_datasets):
        best_idx = np.argmax(means[:, j]) if best == 'max' else np.argmin(means[:, j])
        latex_rows[best_idx][j + 1] = f"\\textbf{{{latex_rows[best_idx][j + 1]}}}"

    # Table header
    col_header = " & ".join(["Model"] + dataset_names) + " \\\\"
    lines = ["\\begin{tabular}{" + "l" + "c" * n_datasets + "}",
             "\\toprule",
             col_header,
             "\\midrule"]
    for row in latex_rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    latex_table = "\n".join(lines)

    # Optional save
    if save_dir and table_name:
        save_dir = os.path.join(FIG_DIR, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, table_name + '.tex'), 'w') as f:
            f.write(latex_table)

    return latex_table

def summarize_model_performance_markdown(data, model_names, dataset_names, precision=3, save_dir=None, table_name=None, best='max'):
    n_models = len(model_names)
    n_datasets = len(dataset_names)

    # Compute mean and 95% CI
    means = np.zeros((n_models, n_datasets))
    cis = np.zeros((n_models, n_datasets))

    for i in range(n_models):
        for j in range(n_datasets):
            values = np.array(data[i][j])
            n = len(values)
            means[i, j] = np.mean(values)

            if n >= 1:
                std_err = sem(values)
                cis[i, j] = std_err * t.ppf(0.975, df=n - 1)
            else:
                raise ValueError(f"Not enough data points for model {model_names[i]} on dataset {dataset_names[j]}.")

    # Build rows
    markdown_rows = []
    for i, model in enumerate(model_names):
        row = [model]
        for j in range(n_datasets):
            mean = means[i, j]
            ci = cis[i, j]
            entry = f"{mean:.{precision}f} ±{ci:.{precision}f}"
            row.append(entry)
        markdown_rows.append(row)

    # Bold best model per column
    for j in range(n_datasets):
        best_idx = np.argmax(means[:, j]) if best == 'max' else np.argmin(means[:, j])
        markdown_rows[best_idx][j + 1] = f"**{markdown_rows[best_idx][j + 1]}**"

    # Create markdown table
    header = ["Model"] + dataset_names
    divider = ["---"] * (n_datasets + 1)
    table_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(divider) + " |"
    ]
    for row in markdown_rows:
        table_lines.append("| " + " | ".join(row) + " |")

    markdown_table = "\n".join(table_lines)

    # Optional save
    if save_dir and table_name:
        save_dir = os.path.join(FIG_DIR, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, table_name + '.md'), 'w') as f:
            f.write(markdown_table)

    return markdown_table