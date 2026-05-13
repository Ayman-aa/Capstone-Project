"""
analyze_results.py

Post-experiment statistical analysis and visualization utilities.

Usage (example):
python analyze_results.py --input experiment_results/blind_scoring_*.csv --output analysis_output

Supports:
- loading long-format scored CSVs (columns: model, prompt_id, rater, score)
- descriptive stats per model
- Cohen's kappa (pairwise raters)
- Kruskal-Wallis across models (non-parametric)
- Pairwise Mann-Whitney U with Holm correction
- boxplot and heatmap of pairwise p-values

"""
from pathlib import Path
import argparse
import glob
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests

sns.set(style="whitegrid")


def load_scored_data(pattern_or_path):
    paths = []
    p = Path(pattern_or_path)
    if p.exists() and p.is_file():
        paths = [str(p)]
    else:
        # treat as glob
        paths = glob.glob(pattern_or_path)

    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern_or_path}")

    dfs = []
    for fp in paths:
        dfs.append(pd.read_csv(fp))
    df = pd.concat(dfs, ignore_index=True)
    return df


def summarize_by_model(df, score_col="score", model_col="model"):
    grp = df.groupby(model_col)[score_col]
    summary = grp.agg(["count", "mean", "median", "std"]).reset_index()
    summary = summary.rename(columns={"count": "n", "std": "sd"})
    return summary


def pivot_prompt_scores(df, model_col="model", prompt_col="prompt_id", rater_col="rater", score_col="score"):
    """Return DataFrame indexed by (model, prompt_id) with raters as columns (wide format)."""
    required = {model_col, prompt_col, rater_col, score_col}
    if not required.issubset(df.columns):
        raise KeyError(f"Dataframe must contain columns: {required}")

    wide = df.pivot_table(index=[model_col, prompt_col], columns=rater_col, values=score_col, aggfunc="first")
    wide = wide.reset_index()
    return wide


def calculate_pairwise_cohen_kappa(df, rater_cols):
    """Compute pairwise Cohen's kappa for provided rater columns.
    Returns DataFrame of kappas with rater pairs.
    """
    rows = []
    for i in range(len(rater_cols)):
        for j in range(i + 1, len(rater_cols)):
            a = df[rater_cols[i]]
            b = df[rater_cols[j]]
            mask = a.notna() & b.notna()
            if mask.sum() == 0:
                k = np.nan
            else:
                k = cohen_kappa_score(a[mask], b[mask])
            rows.append({"rater_a": rater_cols[i], "rater_b": rater_cols[j], "kappa": k, "n": int(mask.sum())})
    return pd.DataFrame(rows)


def kruskal_wallis_on_prompts(df_wide, model_col="model"):
    """Perform Kruskal-Wallis on per-prompt averaged scores across raters.
    df_wide: output of pivot_prompt_scores
    """
    # compute average across rater columns
    rater_cols = [c for c in df_wide.columns if c not in [model_col, "prompt_id"]]
    df_wide = df_wide.copy()
    df_wide["mean_score"] = df_wide[rater_cols].mean(axis=1, skipna=True)

    groups = [g["mean_score"].dropna().values for _, g in df_wide.groupby(model_col)]
    if len(groups) < 2:
        return None
    stat, p = stats.kruskal(*groups)
    return {"statistic": float(stat), "pvalue": float(p)}


def pairwise_mannwhitney(df_wide, model_col="model", correction_method="holm"):
    """Pairwise Mann-Whitney U tests between models using per-prompt mean scores.
    Returns DataFrame with raw and corrected p-values.
    """
    rater_cols = [c for c in df_wide.columns if c not in [model_col, "prompt_id"]]
    df_wide = df_wide.copy()
    df_wide["mean_score"] = df_wide[rater_cols].mean(axis=1, skipna=True)

    models = sorted(df_wide[model_col].unique())
    results = []
    pvals = []
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = df_wide[df_wide[model_col] == models[i]]["mean_score"].dropna().values
            b = df_wide[df_wide[model_col] == models[j]]["mean_score"].dropna().values
            if len(a) < 1 or len(b) < 1:
                p = np.nan
                stat = np.nan
            else:
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            results.append({"model_a": models[i], "model_b": models[j], "statistic": float(stat) if not np.isnan(stat) else None, "pvalue": float(p) if not np.isnan(p) else np.nan})
            pairs.append((models[i], models[j]))
            pvals.append(p if not np.isnan(p) else 1.0)

    # Multiple comparisons correction
    if pvals:
        reject, pvals_corrected, _, _ = multipletests(pvals, method=correction_method)
    else:
        pvals_corrected = []
        reject = []

    for idx, rc in enumerate(results):
        rc["pvalue_corrected"] = float(pvals_corrected[idx]) if pvals_corrected else np.nan
        rc["reject_null"] = bool(reject[idx]) if reject else False

    return pd.DataFrame(results)


def plot_boxplots(df, output_path, score_col="score", model_col="model"):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=model_col, y=score_col, data=df, palette="Set2")
    ax.set_title("Score distribution by model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pairwise_pvalue_heatmap(pairwise_df, output_path):
    # build matrix of corrected p-values
    models = sorted(set(pairwise_df["model_a"]).union(pairwise_df["model_b"]))
    mat = pd.DataFrame(np.ones((len(models), len(models))), index=models, columns=models)
    for _, row in pairwise_df.iterrows():
        a = row["model_a"]
        b = row["model_b"]
        mat.loc[a, b] = row["pvalue_corrected"]
        mat.loc[b, a] = row["pvalue_corrected"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "corrected p-value"})
    plt.title("Pairwise Mann-Whitney U corrected p-values")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_summary_tables(outdir: Path, summary_df, kappa_df=None, kruskal=None, pairwise_df=None):
    outdir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(outdir / "model_summary.csv", index=False)
    if kappa_df is not None:
        kappa_df.to_csv(outdir / "cohen_kappa_pairwise.csv", index=False)
    if kruskal is not None:
        with open(outdir / "kruskal_summary.json", "w") as f:
            json.dump(kruskal, f, indent=2)
    if pairwise_df is not None:
        pairwise_df.to_csv(outdir / "pairwise_mannwhitney.csv", index=False)


def analyze(input_pattern, output_dir, score_col, model_col, rater_col, prompt_col):
    outdir = Path(output_dir)
    df = load_scored_data(input_pattern)

    # If data is long-format with a score column and rater column, proceed.
    summary = summarize_by_model(df, score_col=score_col, model_col=model_col)

    # pivot to wide format for per-prompt analyses
    df_wide = pivot_prompt_scores(df, model_col=model_col, prompt_col=prompt_col, rater_col=rater_col, score_col=score_col)

    # Cohen's kappa pairwise across raters, aggregated across all models/prompts
    rater_cols = [c for c in df_wide.columns if c not in [model_col, prompt_col]]
    kappa_df = None
    if len(rater_cols) >= 2:
        kappa_df = calculate_pairwise_cohen_kappa(df_wide[rater_cols], rater_cols)

    # Kruskal-Wallis across models
    kruskal = kruskal_wallis_on_prompts(df_wide, model_col=model_col)

    # Pairwise Mann-Whitney
    pairwise_df = pairwise_mannwhitney(df_wide, model_col=model_col)

    # Save tables
    save_summary_tables(outdir, summary, kappa_df=kappa_df, kruskal=kruskal, pairwise_df=pairwise_df)

    # Plots
    try:
        plot_boxplots(df, outdir / "boxplot_scores_by_model.png", score_col=score_col, model_col=model_col)
        plot_pairwise_pvalue_heatmap(pairwise_df, outdir / "pairwise_pvalues_heatmap.png")
    except Exception as e:
        print(f"Warning: plotting failed: {e}")

    print(f"Analysis complete. Outputs in: {outdir}")
    return {
        "summary": summary,
        "kappa": kappa_df,
        "kruskal": kruskal,
        "pairwise": pairwise_df,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Analyze blind scoring results from experiments.")
    p.add_argument("--input", required=True, help="Input CSV path or glob (e.g. 'experiment_results/blind_scoring_*.csv')")
    p.add_argument("--output", required=True, help="Output directory for summary and plots")
    p.add_argument("--score-col", default="score", help="Name of the score column")
    p.add_argument("--model-col", default="model", help="Name of the model column")
    p.add_argument("--rater-col", default="rater", help="Name of the rater column")
    p.add_argument("--prompt-col", default="prompt_id", help="Name of the prompt id column")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.input, args.output, args.score_col, args.model_col, args.rater_col, args.prompt_col)
