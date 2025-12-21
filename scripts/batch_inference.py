import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from scripts.feature_engineering import feature_engineering
 
def batch_inference(model, df, save_path=None):
    X, _, feature_cols = feature_engineering(df, fit=False, save=False, is_train=False)
    preds = model.predict(X)
    df['prediction'] = preds
    df['timestamp'] = datetime.now()
 
    if save_path:
        if os.path.isdir(save_path):
            date_str = datetime.now().strftime("%Y%m%d")
            save_file = os.path.join(save_path, f"{date_str}_predictions.csv")
        else:
            save_file = save_path
        df.to_csv(save_file, index=False)
        print(f"Predictions saved to {save_file}")
 
    return df