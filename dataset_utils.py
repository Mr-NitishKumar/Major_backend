import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_dataset(file_path):
    """Analyze the dataset and return basic information."""
    df = pd.read_csv(file_path)
    return {
        "num_features": len(df.columns),
        "num_rows": len(df),
        "feature_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "five_rows": df.sample(5).values.tolist()
    }

def preprocess_dataset(df, impute_missing=False, normalize=False):
    """Preprocess the dataset based on user options."""
    if impute_missing:
        df = df.fillna(df.mean())  # Example: Fill missing values with the mean
    if normalize:
        df = (df - df.mean()) / df.std()  # Example: Standardization
    return df





def handle_missing_values(df, strategy='mean'):
    """Handle missing values in the dataset."""
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Invalid strategy for handling missing values")

def scale_features(df, method='standard'):
    """Scale features using standardization or Min-Max scaling."""
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['number'])
    scaled_data = scaler.fit_transform(numeric_columns)
    df[numeric_columns.columns] = scaled_data
    return df

def encode_categorical_data(df, encoding='onehot'):
    """Encode categorical features in the dataset."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    if encoding == 'onehot':
        return pd.get_dummies(df, columns=categorical_columns)
    elif encoding == 'label':
        encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = encoder.fit_transform(df[col])
        return df
    else:
        raise ValueError("Invalid encoding method")

def split_dataset(df, target_column, test_size=0.2):
    """Split the dataset into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)



def descriptive_statistics(df):
    """Generate descriptive statistics for the dataset."""
    return df.describe().to_dict()

def generate_histograms(df, output_dir="static/histograms"):
    """Generate histograms for numeric features."""
    numeric_columns = df.select_dtypes(include=['number']).columns
    histograms = {}
    for col in numeric_columns:
        plt.figure()
        df[col].plot(kind='hist', bins=20, title=f"Histogram of {col}")
        filename = f"{output_dir}/{col}_histogram.png"
        plt.savefig(filename)
        histograms[col] = filename
        plt.close()
    return histograms

def correlation_heatmap(df, output_file="static/correlation_heatmap.png"):
    """Generate a heatmap of correlations between numerical features."""
    correlation = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(output_file)
    plt.close()
    return output_file

def missing_value_summary(df):
    """Summarize missing values in the dataset."""
    return df.isnull().sum().to_dict()

def categorical_summary(df):
    """Summarize categorical features with frequency counts."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    return {col: df[col].value_counts().to_dict() for col in categorical_columns}
