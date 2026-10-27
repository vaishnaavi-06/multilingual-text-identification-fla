
import re

PROJECT_TITLE = "Multilingual Text Identification Using Finite Automata"

def normalize_text(text):
    """Convert text to lowercase and remove extra spaces."""
    return re.sub(r"\s+", " ", text.strip().lower())

def detect_script(text):
    """Identify common writing systems using Unicode ranges."""
    scripts = {
        "Tamil": (0x0B80, 0x0BFF),
        "Devanagari": (0x0900, 0x097F),
        "Arabic": (0x0600, 0x06FF),
        "Cyrillic": (0x0400, 0x04FF),
        "Greek": (0x0370, 0x03FF),
        "Latin": (0x0041, 0x007A),
    }

    counts = {name: 0 for name in scripts}

    for character in text:
        code = ord(character)
        for name, (start, end) in scripts.items():
            if start <= code <= end:
                counts[name] += 1
                break

    if not any(counts.values()):
        return "Unknown"

    return max(counts, key=counts.get)


def main():
    print("=" * 55)
    print(PROJECT_TITLE)
    print("=" * 55)

    text = input("Enter a sentence in any supported script: ")

    normalized = normalize_text(text)
    script = detect_script(normalized)

    print("\nNormalized text:", normalized)
    print("Detected writing system:", script)


if __name__ == "__main__":
    main()

import pandas as pd

df = pd.read_csv("dataset.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

print("\nNumber of languages:", df["language"].nunique())
print("\nLanguages in dataset:")
print(sorted(df["language"].unique()))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Prepare the dataset
X = df["Text"].astype(str)
y = df["language"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build the language identification model
model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5))),
    ("classifier", LinearSVC())
])

# Train the model
print("\nTraining the language identification model...")
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions))

# Predict the language of a new sentence
print("\n--- Language Prediction ---")
user_text = input("Enter a sentence: ")

predicted_language = model.predict([user_text])[0]
 
print("Predicted language:", predicted_language)


