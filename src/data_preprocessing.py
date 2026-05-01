import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    return X_res, y_res