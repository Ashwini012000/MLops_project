import pandas as pd
import os

class DataProcessor:
    def __init__(self, df, save_path):
        self.df = df.copy()
        self.numeric_cols = None
        self.save_path = save_path

    def clean_data(self):
        # Fill missing student names
        self.df["Name"] = self.df["Name"].fillna("Unknown_Student")

        # Automatically detect numeric columns (exclude Name, Grade, Expected)
        numeric_cols = [col for col in self.df.columns if col not in ["Name", "Grade", "Expected"]]

        # Convert numeric columns to int and fill NaN with 0
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(int)

        # Calculate Total
        self.df["Total"] = self.df[numeric_cols].sum(axis=1)

        self.numeric_cols = numeric_cols + ["Total"]
        self.df["Percentage"] = self.df[["Math", "Science", "English", "History"]].mean(axis=1)

        # numeric_cols = [col for col in self.df.columns if col not in ["Name", "Grade", "Expected"]]
        # self.df["HasFailedSubject"] = (self.df[numeric_cols[:-1]] < 35).any(axis=1).astype(int)

        return self.df

        
    def assign_grade(self):
    # Default grade
     subject_cols = self.numeric_cols[:-1]      # All subject columns except Total
     max_marks = len(subject_cols) * 100        # Example: 4 subjects → 400
     passing_total = max_marks * 0.40           # 40% required to pass

    # ✔ Add Percentage Column
     self.df["Percentage"] = (self.df["Total"] / max_marks) * 100

    # Default Grade
     self.df["Grade"] = "D"

    # 1️⃣ Fail if student did NOT attend exam (any subject is NaN)
     self.df.loc[self.df[subject_cols].isna().any(axis=1), "Grade"] = "Fail"

    # 2️⃣ Fail if any subject < 35
     self.df.loc[(self.df[subject_cols] < 35).any(axis=1), "Grade"] = "Fail"

    # 3️⃣ Fail if Total < passing marks
     self.df.loc[(self.df["Total"] < passing_total), "Grade"] = "Fail"

    # Apply percentage grading only for students NOT failed
     self.df.loc[(self.df["Percentage"] >= 85) & (self.df["Grade"] != "Fail"), "Grade"] = "A"
     self.df.loc[(self.df["Percentage"] >= 70) & (self.df["Percentage"] < 85) & (self.df["Grade"] != "Fail"), "Grade"] = "B"
     self.df.loc[(self.df["Percentage"] >= 50) & (self.df["Percentage"] < 70) & (self.df["Grade"] != "Fail"), "Grade"] = "C"
     self.df.loc[(self.df["Percentage"] >= 35) & (self.df["Percentage"] < 50) & (self.df["Grade"] != "Fail"), "Grade"] = "D"


     return self.df
    
    def save_processed_data(self):
        """Save processed data to CSV inside data/processed/"""
        if self.save_path is None:
            self.save_path = "data/unknown_processed/processed_data.csv"
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.df.to_csv(self.save_path, index=False)
        print(f"Processed data saved successfully at: {self.save_path}")

    def run_pipeline(self):
        """Run the full pipeline and save data"""
        print(" Cleaning data...")
        self.clean_data()

        print(" Assigning grades...")
        self.assign_grade()

        print(" Saving processed data...")
        self.save_processed_data()

        return self.df
