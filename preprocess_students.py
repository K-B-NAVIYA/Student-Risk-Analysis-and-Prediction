# preprocess_students.py
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance_Labeled.csv")

# Display initial info
print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# Rename columns for consistency
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Encode categorical columns
encode_cols = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
df = pd.get_dummies(df, columns=encode_cols, drop_first=True)

# Compute average score (optional feature)
df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)

# Drop missing values if any
df.dropna(inplace=True)

# Save cleaned version
df.to_csv("StudentsPerformance_Cleaned.csv", index=False)

print("✅ Cleaned dataset saved as StudentsPerformance_Cleaned.csv")
print("Final shape:", df.shape)
