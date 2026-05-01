from src.data_preprocessing import load_data, preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model

def main():
    print("🚀 Starting Fraud Detection System...")

    # Load data
    df = load_data("data/creditcard.csv")

    # Preprocess
    X, y = preprocess_data(df)

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    print("✅ Project completed. Check outputs folder.")

if __name__ == "__main__":
    main()