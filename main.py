import argparse
import pandas as pd

from src.preprocess import load_and_prepare
from src.model import train_models
from src.detect import detect_anomalies
from src.explain import explain_anomalies
from src.visualize import plot_scores, plot_top_features
from src.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multivariate Time Series Anomaly Detection (Hackathon Spec)"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--report", action="store_true", help="Generate PDF report")
    parser.add_argument(
        "--timestamp_col",
        default="auto",
        help="Timestamp column name (or 'auto' to autodetect)",
    )
    # Defaults per hackathon spec
    parser.add_argument("--train_start", default="2004-01-01 00:00")
    parser.add_argument("--train_end", default="2004-01-05 23:59")
    parser.add_argument("--analysis_start", default="2004-01-01 00:00")
    parser.add_argument("--analysis_end", default="2004-01-19 07:59")

    # Flex knobs
    parser.add_argument("--perc_threshold", type=float, default=0.97,
                        help="Percentile threshold for anomaly detection (default=0.97)")
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Moving average window for score smoothing (default=5)")
    parser.add_argument("--top_k", type=int, default=7,
                        help="Number of top features to output (default=7)")
    parser.add_argument("--min_contrib", type=float, default=1.0,
                        help="Minimum % contribution to consider (default=1.0)")

    args = parser.parse_args()

    # 1) Load & preprocess
    df, X_full, meta = load_and_prepare(
        csv_path=args.input,
        timestamp_col=args.timestamp_col,
        normal_start=args.train_start,
        normal_end=args.train_end,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
    )

    # 2) Train models
    models = train_models(meta["X_train"])

    # 3) Detect anomalies
    scores, labels, components = detect_anomalies(
        models=models,
        X=X_full,
        meta=meta,
        perc_threshold=args.perc_threshold,
        smooth_window=args.smooth_window,
    )

    # Store calibrated training stats for report consistency
    train_scores = scores[meta["train_mask"]]
    components["train_mean"] = float(train_scores.mean())
    components["train_max"] = float(train_scores.max())

    # 4) Feature attribution
    top_lists = explain_anomalies(
        X=X_full,
        scores=scores,
        labels=labels,
        meta=meta,
        max_k=args.top_k,
        min_percent=args.min_contrib,
    )

    # 5) Build required output (8 columns)
    out = pd.DataFrame()
    out["timestamp"] = meta["timestamps"]
    out["Abnormality_score"] = scores.astype(float)
    for k in range(args.top_k):
        out[f"top_feature_{k+1}"] = [lst[k] if k < len(lst) else "" for lst in top_lists]

    # convenience only
    out["Is_Anomaly"] = labels.astype(int)

    out.to_csv(args.output, index=False)
    print(f"✅ Results saved → {args.output}")

    # 6) Compliance check (console)
    pass_fail = "✅ PASS" if (components["train_mean"] < 10 and components["train_max"] < 25) else "❌ FAIL"
    print(
        f"Training window score stats → mean={components['train_mean']:.2f}, "
        f"max={components['train_max']:.2f}   → {pass_fail}"
    )

    # 7) Plots
    if args.plot:
        plot_scores(
            meta["timestamps"],
            scores,
            labels,
            threshold=components.get("threshold", 30.0)  # use real threshold
        )
        plot_top_features(out, labels, meta["feature_cols"], top_n=5)

    # 8) PDF Report
    if args.report:
        pdf_path = args.output.rsplit(".", 1)[0] + ".pdf"
        generate_report(
            df_with_scores=out,
            pdf_path=pdf_path,
            feature_cols=meta["feature_cols"],
            meta=meta,
            model_components=components,
        )
        print(f"📄 Report generated → {pdf_path}")


if __name__ == "__main__":
    main()
