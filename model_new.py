import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).with_name("mail_data.csv")
MODEL_PATH = Path(__file__).with_name("spam_sms_model.pkl")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_evaluate() -> tuple[TfidfVectorizer, LogisticRegression]:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Class": "label", "Message": "message"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, target_names=["ham", "spam"]))

    with MODEL_PATH.open("wb") as handle:
        pickle.dump((vectorizer, model), handle)

    return vectorizer, model


def predict_message(message: str) -> str:
    with MODEL_PATH.open("rb") as handle:
        vectorizer, model = pickle.load(handle)

    cleaned = clean_text(message)
    prediction = model.predict(vectorizer.transform([cleaned]))[0]
    return "spam" if prediction == 1 else "ham"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test a spam SMS classifier")
    parser.add_argument("--message", type=str, help="A message to classify")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        train_and_evaluate()

    if args.message:
        print(f"Prediction: {predict_message(args.message)}")
    else:
        sample_messages = [
            "Congratulations! You have won a free prize.",
            "Hi, are we still meeting tomorrow?",
            "Free entry to win a cash prize now",
        ]
        for sample in sample_messages:
            print(f"{sample} -> {predict_message(sample)}")


if __name__ == "__main__":
    main()
