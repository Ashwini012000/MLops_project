import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class ModelTrainer:
    """Train multiple ML models on student dataset and save all models."""

    def __init__(self, df, target_col="Grade", model_dir="models"):
        self.df = df.copy()
        self.target_col = target_col
        self.model_dir = model_dir
        self.label_encoder = LabelEncoder()
        self.models = {
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
            "SVM": SVC(kernel="linear", probability=True, random_state=42)
        }
        self.results = {}
        self.best_model_name = None
        self.feature_cols = None

        # Create directory to save models
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_data(self):
     """Prepare X (features) and y (target)."""
    
     numeric_cols = [col for col in self.df.columns if col not in ["Name", self.target_col]]
     self.feature_cols = numeric_cols

     X = self.df[self.feature_cols]
     #here grade is converted to numbers
     y = self.label_encoder.fit_transform(self.df[self.target_col])
     #print(f"target:{y}")

     print("\nINPUT FEATURES (X):")
     print(X.head())

     print("\nTARGET LABELS (y):")
     print(self.df[self.target_col].head())

     return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self):
        """Train all models and save them."""
        X_train, X_test, y_train, y_test = self.prepare_data()
        print(f"input columns:{X_train}")
        print(f"target column:{y_train}")
        

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.results[name] = acc

            # Save each model in models/ directory
            model_path = os.path.join(self.model_dir, f"{name}_model.joblib")
            joblib.dump({
                "model": model,
                "label_encoder": self.label_encoder,
                "feature_cols": self.feature_cols,
                "model_name": name,
                "accuracy": acc
            }, model_path)

            print(f" Saved: {name}  {model_path} | Accuracy: {acc*100:.2f}%")

        # Just print best model — do not save it separately
        self.best_model_name = max(self.results, key=self.results.get)
        print(f"\n Best Model: {self.best_model_name} ({self.results[self.best_model_name]*100:.2f}%)")
