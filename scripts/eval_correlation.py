import json
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def normalize_path(p: str) -> str:
    if p is None:
        return ""
    return str(p).replace("\\", "/").strip()


def make_summary_key(p: str) -> str:
    """
    Match paths by the portion after summary_clean/.
    This avoids mismatch between absolute and relative paths.
    """
    p = normalize_path(p)

    marker = "summary_clean/"
    if marker in p:
        return p.split(marker, 1)[1]

    parts = p.split("/")
    return "/".join(parts[-4:])


def load_json_results(json_path: str):
    """
    Load results from:
    1. JSON file containing list[dict]
    2. JSON file containing one dict
    3. JSONL file
    4. directory of JSON files
    """
    path = Path(json_path)
    records = []

    if path.is_dir():
        for file in path.rglob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    records.append(data)

    elif path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
            else:
                raise ValueError("JSON file must contain either a dict or a list of dicts.")

    return records


def compute_pairwise_correlations(df, score_cols):
    rows = []

    for i in range(len(score_cols)):
        for j in range(i + 1, len(score_cols)):
            x_col = score_cols[i]
            y_col = score_cols[j]

            sub = df[[x_col, y_col]].dropna()

            if len(sub) < 2:
                rows.append({
                    "score_1": x_col,
                    "score_2": y_col,
                    "n": len(sub),
                    "pearson_r": None,
                    "pearson_p": None,
                    "spearman_rho": None,
                    "spearman_p": None,
                })
                continue

            pearson_r, pearson_p = pearsonr(sub[x_col], sub[y_col])
            spearman_rho, spearman_p = spearmanr(sub[x_col], sub[y_col])

            rows.append({
                "score_1": x_col,
                "score_2": y_col,
                "n": len(sub),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            })

    return pd.DataFrame(rows)


def main(csv_file, faithfulness_json, coverage_json, output_file=None):
    # -------------------------
    # Load CSV: SummaC results
    # -------------------------
    csv_df = pd.read_csv(csv_file)

    required_csv_cols = {"summac_conv_score", "sum_path"}
    missing = required_csv_cols - set(csv_df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    csv_df["match_key"] = csv_df["sum_path"].apply(make_summary_key)

    csv_scores = csv_df[
        [
            "match_key",
            "corpus",
            "rel_path",
            "sum_path",
            "summac_conv_score",
        ]
    ].copy()

    # -------------------------
    # Load JSON: faithfulness
    # -------------------------
    faithfulness_records = load_json_results(faithfulness_json)
    faithfulness_df = pd.DataFrame(faithfulness_records)

    required_faithfulness_cols = {"summary_file", "faithfulness_score"}
    missing = required_faithfulness_cols - set(faithfulness_df.columns)
    if missing:
        raise ValueError(f"Faithfulness JSON is missing required fields: {missing}")

    faithfulness_df["match_key"] = faithfulness_df["summary_file"].apply(make_summary_key)

    faithfulness_scores = faithfulness_df[
        [
            "match_key",
            "faithfulness_score",
            "total_claims",
            "supported_claims",
            "partially_supported_claims",
            "unsupported_claims",
        ]
    ].copy()

    # -------------------------
    # Load JSON: coverage
    # -------------------------
    coverage_records = load_json_results(coverage_json)
    coverage_df = pd.DataFrame(coverage_records)

    required_coverage_cols = {"summary_file", "coverage_score"}
    missing = required_coverage_cols - set(coverage_df.columns)
    if missing:
        raise ValueError(f"Coverage JSON is missing required fields: {missing}")

    coverage_df["match_key"] = coverage_df["summary_file"].apply(make_summary_key)

    coverage_scores = coverage_df[
        [
            "match_key",
            "coverage_score",
            "total_points",
            "included_points",
            "partially_included_points",
            "not_included_points",
        ]
    ].copy()

    # -------------------------
    # Debug matching
    # -------------------------
    csv_keys = set(csv_scores["match_key"])
    faithfulness_keys = set(faithfulness_scores["match_key"])
    coverage_keys = set(coverage_scores["match_key"])

    print(f"CSV rows: {len(csv_scores)}")
    print(f"Faithfulness JSON rows: {len(faithfulness_scores)}")
    print(f"Coverage JSON rows: {len(coverage_scores)}")

    print()
    print(f"CSV ∩ Faithfulness: {len(csv_keys & faithfulness_keys)}")
    print(f"CSV ∩ Coverage: {len(csv_keys & coverage_keys)}")
    print(f"Faithfulness ∩ Coverage: {len(faithfulness_keys & coverage_keys)}")
    print(f"All three overlap: {len(csv_keys & faithfulness_keys & coverage_keys)}")

    # -------------------------
    # Merge all three
    # -------------------------
    merged = (
        csv_scores
        .merge(faithfulness_scores, on="match_key", how="inner")
        .merge(coverage_scores, on="match_key", how="inner")
    )

    print()
    print(f"Matched rows across all three files: {len(merged)}")

    if len(merged) == 0:
        print("No rows matched across all three files.")
        return

    # Convert scores to numeric
    score_cols = [
        "summac_conv_score",
        "faithfulness_score",
        "coverage_score",
    ]

    for col in score_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    analysis_df = merged.dropna(subset=score_cols).copy()

    print(f"Rows usable for all three-score correlation: {len(analysis_df)}")

    if len(analysis_df) < 2:
        print("Need at least 2 valid rows to compute correlations.")
        return

    # -------------------------
    # Pairwise correlations
    # -------------------------
    corr_df = compute_pairwise_correlations(analysis_df, score_cols)

    print()
    print("Pairwise correlation results")
    print("----------------------------")
    print(corr_df.to_string(index=False))

    print()
    print("Score summaries")
    print("---------------")
    print(analysis_df[score_cols].describe())

    # Optional save
    if output_file:
        analysis_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        corr_output = Path(output_file).with_name(
            Path(output_file).stem + "_correlations.csv"
        )
        corr_df.to_csv(corr_output, index=False, encoding="utf-8-sig")

        print()
        print(f"Merged analysis file saved to: {output_file}")
        print(f"Correlation table saved to: {corr_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correlate SummaCConv, faithfulness, and coverage scores."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV file containing summac_conv_score and sum_path."
    )

    parser.add_argument(
        "--faith",
        required=True,
        help="JSON/JSONL file or directory containing faithfulness_score results."
    )

    parser.add_argument(
        "--cov",
        required=True,
        help="JSON/JSONL file or directory containing coverage_score results."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save merged analysis data."
    )

    args = parser.parse_args()

    main(
        csv_file=args.csv,
        faithfulness_json=args.faith,
        coverage_json=args.cov,
        output_file=args.output,
    )