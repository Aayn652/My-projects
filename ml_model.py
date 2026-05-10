import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ✅ Step 1: Set working directory
os.chdir(r"C:\Users\ASUS\OneDrive\Desktop\OLYMPIC DATA ANALYSIS")

# ✅ Step 2: Load datasets
df = pd.read_csv(r"C:\Users\ASUS\OneDrive\ドキュメント\OLYMPIC DATA ANALYSIS\olympics-data-analysis-web-app\athlete_events.csv")

region_df = pd.read_csv(r"C:\Users\ASUS\OneDrive\ドキュメント\OLYMPIC DATA ANALYSIS\olympics-data-analysis-web-app\noc_regions.csv")


# ✅ Step 3: Merge datasets
df = pd.merge(df, region_df, on='NOC', how='left')

# ✅ Step 4: Clean data
df = df.dropna(subset=['Medal', 'Age', 'Height', 'Weight', 'Sex', 'Sport'])

# ✅ Step 5: Encode categorical values
le_sex = LabelEncoder()
le_sport = LabelEncoder()
le_medal = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Sport'] = le_sport.fit_transform(df['Sport'])
df['Medal'] = le_medal.fit_transform(df['Medal'])

# ✅ Step 6: Features & Target
X = df[['Age', 'Height', 'Weight', 'Sex', 'Sport']]
y = df['Medal']

# ✅ Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 8: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Step 9: Save model and encoders
pickle.dump((model, le_sex, le_sport, le_medal), open('model.pkl', 'wb'))

print("✅ Model trained successfully and saved as model.pkl")
