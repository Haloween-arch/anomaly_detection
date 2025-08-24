# 📌 Multivariate Time Series Anomaly Detection

# 📝 Project Description
overview: |
  This project implements a hybrid anomaly detection pipeline
  for multivariate time series data, designed for hackathon specifications.
  It combines Isolation Forest, PCA reconstruction error, and rolling change
  magnitude into a unified anomaly score (0–100).  
  The system ensures compliance with rules:
    - Training scores mean < 10
    - Training scores max < 25  

# 📂 Folder Structure
structure:
  - main.py               # Entry point (CLI script)
  - requirements.txt      # Python dependencies
  - src/
      - preprocess.py     # Data loading & preprocessing
      - model.py          # Train models (IsolationForest, PCA)
      - detect.py         # Hybrid anomaly scoring
      - explain.py        # Feature attribution (top contributors)
      - visualize.py      # Score + feature plots
      - report.py         # PDF report generation
  - data/
      - input.csv         # Example input dataset
  - output/
      - anomalies.csv     # Generated anomaly results
      - anomalies.pdf     # Generated anomaly report

# Requirements
Dependencies listed in `requirements.txt`:

pandas>=2.0.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
reportlab>=3.6.12 

# ⚙️ Environment Setup
steps:
  - Create a virtual environment:
      windows: python -m venv venv
      linux/mac: python3 -m venv venv
  - Activate environment:
      windows: .\venv\Scripts\activate
      linux/mac: source venv/bin/activate
  - Install dependencies:
      command: pip install -r requirements.txt
  - (Optional) Save dependencies:
      command: pip freeze > requirements.txt

# ▶️ Running the Pipeline
- example_command: 
  python main.py --input data/input.csv --output output/anomalies.csv --plot --report --timestamp_col auto --train_start "2004-01-01 00:00" --train_end "2004-01-05 23:59" --analysis_start "2004-01-01 00:00" --analysis_end "2004-01-19 07:59"

# 🎯 Key Arguments
flags:
  - input: "Path to input CSV"
  - output: "Path to output CSV"
  - plot: "Show plots interactively"
  - report: "Generate PDF report"
  - timestamp_col: "Specify timestamp column (or auto)"
  - train_start / --train_end: "Training period (normal baseline)"
  - analysis_start / --analysis_end: "Full analysis window"
  - perc_threshold: "Percentile threshold for anomaly detection (default 0.97)"
  - smooth_window: "Smoothing window size (default 5)"
  - top_k: "Number of top features (default 7)"
  - min_contrib: "Minimum % contribution (default 1.0)"

# 📊 Outputs
- anomalies.csv → with columns:
    - timestamp
    - Abnormality_score (0–100)
    - top_feature_1 ... top_feature_7
    - Is_Anomaly (0/1 flag)
- anomalies.pdf → judge-friendly summary report:
    - Executive summary (training stats, threshold, compliance)
    - Anomaly score timeline with threshold
    - Component plots (IsolationForest, PCA, rolling changes)
    - Top contributing features bar chart
- Optional plots (when --plot used):
    - Score timeline
    - Feature contribution distribution

# 🧪 Methodology
pipeline:
  - Preprocessing: standardize features, split train/analysis windows
  - Modeling:
      - IsolationForest → capture point anomalies
      - PCA → capture correlation shifts
      - Rolling norm → capture sudden temporal changes
  - Hybrid Score: weighted average (0.5 * IF + 0.35 * PCA + 0.15 * Roll)
  - Calibration: enforce hackathon compliance (mean<10, max<25 in train window)
  - Labeling: threshold by percentile (default 97th)

# ⚠️ Hackathon Compliance
rules:
  - Training mean < 10
  - Training max < 25
  - Report clearly displays PASS/FAIL
  - Threshold is percentile-based (judge friendly)
  - PDF report includes:
      - Analysis window
      - Training window
      - Detected anomaly count
      - Compliance stats
      - Contributing features

# 🛠️ Notes
- Adjust `--perc_threshold` or `--smooth_window` if compliance fails.
- Virtual environment recommended to avoid dependency conflicts.
- Judges can re-run with different params for sensitivity testing.
