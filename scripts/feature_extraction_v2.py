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
# Feature extraction (updated for short series)
# ------------------------------------------------------------
def calculate_seasonality_trend(series, period):
    decomposition = seasonal_decompose(series, model='additive', period=period)
    return decomposition.trend, decomposition.seasonal, decomposition.resid

def calculate_strength(trend, seasonal, residual):
    trend_strength = 1 - (residual.var() / (trend + residual).var())
    seasonal_strength = 1 - (residual.var() / (seasonal + residual).var())
    return trend_strength, seasonal_strength

def calculate_autocorrelation(series):
    if len(series) < 2:
        return np.nan
    return acf(series, nlags=1)[1]

def calculate_stationarity(series):
    if len(series) < 3:
        return np.nan
    return adfuller(series)[1] < 0.05

def calculate_additional_features(series):
    n = len(series)
    if n == 0:
        return {k: np.nan for k in ["Mean", "Variance", "Skewness", "Kurtosis", "Peak-to-Peak", "Energy", "Entropy"]}
    mean_val = np.mean(series)
    var_val = np.var(series)
    skew_val = skew(series) if n >= 3 else np.nan
    kurt_val = kurtosis(series) if n >= 4 else np.nan
    peak_to_peak = np.ptp(series)
    energy_val = np.sum(series**2)
    # Entropy: need at least 2 distinct values
    if n >= 2 and len(np.unique(series)) >= 2:
        entropy_val = entropy(pd.Series(series).value_counts().values)
    else:
        entropy_val = np.nan
    return {
        "Mean": mean_val,
        "Variance": var_val,
        "Skewness": skew_val,
        "Kurtosis": kurt_val,
        "Peak-to-Peak": peak_to_peak,
        "Energy": energy_val,
        "Entropy": entropy_val
    }

def get_time_series_features(series, period=12):
    n = len(series)
    features = calculate_additional_features(series)
    features["Autocorrelation (Lag-1)"] = calculate_autocorrelation(series)
    features["Stationarity (ADF Test)"] = calculate_stationarity(series)
    
    if n >= 2 * period:
        try:
            trend, seasonal, residual = calculate_seasonality_trend(series, period)
            t_strength, s_strength = calculate_strength(trend, seasonal, residual)
            features["Trend Strength"] = t_strength
            features["Seasonality Strength"] = s_strength
        except:
            features["Trend Strength"] = np.nan
            features["Seasonality Strength"] = np.nan
    else:
        features["Trend Strength"] = np.nan
        features["Seasonality Strength"] = np.nan
    return features

# ------------------------------------------------------------
# Loaders (same as before, but ensure they return numpy arrays)
# ------------------------------------------------------------
def is_numeric_string(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_file_with_fallback_encoding(file_path):
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.readlines()

def load_ucr_ts(file_path):
    lines = read_file_with_fallback_encoding(file_path)
    numbers = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        tokens = line.split()
        nums = [float(t) for t in tokens if is_numeric_string(t)]
        numbers.extend(nums)
    return np.array(numbers)

def load_tsf(file_path):
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
            values = [p.strip() for p in parts[2:] if is_numeric_string(p.strip())]
            numbers.extend([float(v) for v in values])
    return np.array(numbers)

def load_csv_numeric(file_path):
    try:
        df = pd.read_csv(file_path)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns")
        flat = numeric_df.values.flatten()
        flat = flat[~np.isnan(flat)]
        return flat
    except Exception:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        numbers = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
        return np.array([float(n) for n in numbers])

def load_any_time_series(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.ts':
        return load_ucr_ts(file_path)
    elif ext == '.tsf':
        return load_tsf(file_path)
    elif ext == '.csv':
        return load_csv_numeric(file_path)
    else:
        try:
            return load_ucr_ts(file_path)
        except:
            return load_csv_numeric(file_path)

# ------------------------------------------------------------
# Main processing (no length filter except <3)
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
            if len(series) < 3:
                print(f"  -> SKIPPED: series length {len(series)} < 3 (too short for any feature)")
                errors.append(f"{dataset_name}: length {len(series)} < 3")
                continue
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
period = 12
max_points = 100_000

features_df, errors = generate_features_table_from_local(folder_path, period, max_points)

print("\n" + "="*80)
print("FINAL FEATURES TABLE (INCLUDING SHORT SERIES)")
print("="*80)
print(features_df.to_string())

features_df.to_csv("time_series_features_all_55_datasets.csv")
print("\nTable saved as 'time_series_features_all_55_datasets.csv'")

if errors:
    print("\n" + "="*80)
    print("FILES SKIPPED (length < 3)")
    print("="*80)
    for err in errors:
        print(f"- {err}")