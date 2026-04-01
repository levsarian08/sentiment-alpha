# Telemetry Anomaly Detection

Anomaly detection on real **AWS CloudWatch** server telemetry using **Isolation Forest** and an **LSTM Autoencoder**. Flags abnormal sensor readings in cloud infrastructure streams — directly relevant to network health monitoring, satellite operations, and any system requiring automated anomaly alerting at scale.

Built with the **NAB (Numenta Anomaly Benchmark)** dataset, which includes real-world AWS server metrics with expert-labeled ground-truth anomalies.

---

## Results (AWS CloudWatch, 7 evaluated channels)

| Model              | Mean F1 | Mean AUC |
|--------------------|---------|----------|
| Isolation Forest   |  0.83   |  0.78    |
| LSTM Autoencoder   |  0.78   |  0.70    |

IF outperforms on sudden spike anomalies (e.g. grok_asg_anomaly, F1=1.0 vs 0.32).  
LSTM outperforms on gradual/subtle pattern shifts (e.g. rds_cpu_utilization, F1=1.0 vs 0.53).

> Evaluation uses **windowed F1** (±100 timesteps around each labeled point) — the standard protocol for NAB's sparse single-point labels.

---

## Project Structure

```
anomaly-detection-telemetry/
├── src/
│   ├── data/
│   │   ├── download.py          # load NAB channels + anomaly labels
│   │   └── preprocess.py        # windowing, normalization, train/test splits
│   ├── models/
│   │   ├── isolation_forest.py  # IF wrapper with window features
│   │   └── lstm_autoencoder.py  # PyTorch seq-to-seq autoencoder
│   └── evaluate.py              # windowed F1, threshold tuning, Plotly charts
├── notebooks/
│   └── eda_and_results.ipynb    # EDA + full results walkthrough
├── app/
│   └── dashboard.py             # Streamlit anomaly explorer
├── train.py                     # end-to-end training script
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/levsarian08/anomaly-detection-telemetry.git
cd anomaly-detection-telemetry
pip install -r requirements.txt

# 2. Download the NAB dataset from https://github.com/numenta/NAB
#    and move the folders into place:
mv ~/Downloads/NAB-master/data data/raw/nab
mv ~/Downloads/NAB-master/labels data/raw/labels

# 3. Train both models on a channel
python train.py --channel realAWSCloudwatch/ec2_cpu_utilization_24ae8d

# 4. Train on all channels
python train.py --all-channels --output results/summary.csv

# 5. Launch the interactive dashboard
streamlit run app/dashboard.py
```

---

## Dataset

**NAB (Numenta Anomaly Benchmark)** — a collection of real-world time series from AWS CloudWatch, Twitter, traffic sensors, and more, with expert-labeled anomaly timestamps.

This project uses the **realAWSCloudwatch** subset:
- 17 channels of EC2, RDS, ELB, and network metrics
- 7 channels with evaluable anomalies in the test split
- Labels: single-point timestamps marking anomaly peaks

Source: [Numenta/NAB on GitHub](https://github.com/numenta/NAB)

---

## Approach

### 1. Isolation Forest (Baseline)
Each sliding window is flattened and augmented with statistical features (mean, std, min, max, peak-to-peak, mean absolute delta). Trained on the normal portion of the training split. Anomaly score is the negative mean path length across isolation trees.

### 2. LSTM Autoencoder
A sequence-to-sequence PyTorch model that learns to reconstruct normal windows. At inference, high reconstruction error (MSE) flags anomalies. Threshold is calibrated on training reconstruction errors. Better than IF on gradual drift patterns requiring temporal context.

### Evaluation
Both models use **windowed F1** — a prediction within ±100 timesteps of a labeled anomaly point counts as a true positive. This is the standard protocol for NAB's sparse point labels. AUC-ROC is reported as the primary ranking metric.

---

## Dashboard

Interactive anomaly explorer built with Streamlit. Switch between channels, toggle models, adjust the detection threshold, and compare IF vs LSTM side by side.

![Isolation Forest](screenshots/dashboard_if.png)
![LSTM Autoencoder](screenshots/dashboard_lstm.png)

---

## Tech Stack

`Python` · `PyTorch` · `Scikit-learn` · `Pandas` · `NumPy` · `Streamlit` · `Plotly`
