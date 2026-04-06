import os
import re
import numpy as np
import pandas as pd
import warnings
from scipy.stats import skew, kurtosis, entropy
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Feature extraction (unchanged)
# ------------------------------------------------------------
def calculate_seasonality_trend(series, period):
    decomposition = seasonal_decompose(series, model='additive', period=period)
    return decomposition.trend, decomposition.seasonal, decomposition.resid

def calculate_strength(trend, seasonal, residual):
    trend_strength = 1 - (residual.var() / (trend + residual).var())
    seasonal_strength = 1 - (residual.var() / (seasonal + residual).var())
    return trend_strength, seasonal_strength

def calculate_autocorrelation(series):
    return acf(series, nlags=1)[1]

def calculate_stationarity(series):
    return adfuller(series)[1] < 0.05

def calculate_additional_features(series):
    return {
        "Mean": np.mean(series),
        "Variance": np.var(series),
        "Skewness": skew(series),
        "Kurtosis": kurtosis(series),
        "Peak-to-Peak": np.ptp(series),
        "Energy": np.sum(series**2),
        "Entropy": entropy(series.value_counts().values)
    }

def get_time_series_features(series, period=12):
    if len(series) < 2 * period:
        raise ValueError(f"Series too short: {len(series)} < {2*period}")
    # If series is extremely long, subsample to 100k points for decomposition
    if len(series) > 100_000:
        series = series.iloc[::len(series)//100_000]  # take approx 100k points
    trend, seasonal, residual = calculate_seasonality_trend(series, period)
    t_strength, s_strength = calculate_strength(trend, seasonal, residual)
    features = {
        "Trend Strength": t_strength,
        "Seasonality Strength": s_strength,
        "Autocorrelation (Lag-1)": calculate_autocorrelation(series),
        "Stationarity (ADF Test)": calculate_stationarity(series)
    }
    features.update(calculate_additional_features(series))
    return features

# ------------------------------------------------------------
# Robust loaders with encoding fallback and numeric extraction
# ------------------------------------------------------------
def is_numeric_string(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_file_with_fallback_encoding(file_path):
    """Try utf-8, then latin1, then ignore errors."""
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    # Last resort: open with 'ignore'
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.readlines()

def load_ucr_ts(file_path):
    """
    Load .ts file (UCR format). Each line: label + space-separated numbers.
    Extracts ALL numbers from all lines.
    """
    lines = read_file_with_fallback_encoding(file_path)
    numbers = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Split on whitespace
        tokens = line.split()
        # Extract all tokens that are numeric (this includes the label if it's numeric)
        # But we want the actual time series values, which are all numbers after the label.
        # Simpler: take all tokens that are numeric and ignore non-numeric.
        nums = [float(t) for t in tokens if is_numeric_string(t)]
        numbers.extend(nums)
    return np.array(numbers)

def load_tsf(file_path):
    """Load .tsf file (forecasting format)."""
    lines = read_file_with_fallback_encoding(file_path)
    in_data = False
    numbers = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('@data'):
            in_data = True
            continue
        if line.startswith('@'):
            continue
        if in_data:
            parts = line.split(',')
            # First two parts are series name and start timestamp, rest are values
            values = [p.strip() for p in parts[2:] if is_numeric_string(p.strip())]
            numbers.extend([float(v) for v in values])
    return np.array(numbers)

def load_csv_numeric(file_path):
    """
    Load CSV, handle irregular rows, extract all numeric cells.
    """
    # Try reading with pandas, but if fails, fallback to raw numeric extraction
    try:
        # First attempt: normal read
        df = pd.read_csv(file_path)
        # If the first column contains strings like 'Date', we drop non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns")
        flat = numeric_df.values.flatten()
        flat = flat[~np.isnan(flat)]
        return flat
    except Exception:
        # Fallback: read as raw text and extract all numbers using regex
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        # Find all floating point numbers (including scientific notation)
        numbers = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
        return np.array([float(n) for n in numbers])

def load_any_time_series(file_path):
    """Universal loader based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.ts':
        return load_ucr_ts(file_path)
    elif ext == '.tsf':
        return load_tsf(file_path)
    elif ext == '.csv':
        return load_csv_numeric(file_path)
    else:
        # Unknown: try as .ts first, then as .csv
        try:
            return load_ucr_ts(file_path)
        except:
            return load_csv_numeric(file_path)

# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------
def generate_features_table_from_local(folder_path, period=12, max_points=100_000):
    exclude_extensions = ('.py', '.txt', '.doc', '.docx', '.xlsx', '.xls', '.md', '.pdf', '.jpg', '.png')
    all_files = [f for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f))
                 and not f.lower().endswith(exclude_extensions)
                 and not f.startswith('.')]
    
    features_list = []
    errors = []
    
    for i, filename in enumerate(all_files):
        file_path = os.path.join(folder_path, filename)
        dataset_name = filename
        print(f"Processing [{i+1}/{len(all_files)}]: {dataset_name} ...")
        
        try:
            series = load_any_time_series(file_path)
            print(f"  -> Extracted {len(series)} numeric values")
            if len(series) == 0:
                print(f"  -> SKIPPED: empty series")
                continue
            if len(series) < 2 * period:
                print(f"  -> SKIPPED: series length {len(series)} < {2*period}")
                continue
            # Subsample if too long
            if len(series) > max_points:
                step = len(series) // max_points
                series = series[::step]
                print(f"  -> Subsample to {len(series)} points")
            features = get_time_series_features(pd.Series(series), period)
            features["Dataset"] = dataset_name
            features_list.append(features)
            print(f"  -> OK")
        except Exception as e:
            errors.append(f"{dataset_name}: {str(e)}")
            print(f"  -> FAILED: {e}")
    
    features_df = pd.DataFrame(features_list)
    if not features_df.empty:
        features_df.set_index("Dataset", inplace=True)
    return features_df, errors

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
folder_path = r"E:\\IPS\\Datasets"
period = 12          # adjust if your data has different seasonality (e.g., 7 for daily)
max_points = 100_000 # safe limit for decomposition

features_df, errors = generate_features_table_from_local(folder_path, period, max_points)

print("\n" + "="*80)
print("FINAL FEATURES TABLE")
print("="*80)
print(features_df.to_string())

features_df.to_csv("time_series_features_all_files.csv")
print("\nTable saved as 'time_series_features_all_files.csv'")

if errors:
    print("\n" + "="*80)
    print("ERRORS ENCOUNTERED")
    print("="*80)
    for err in errors:
        print(f"- {err}")