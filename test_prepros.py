from src.preprocessing import load_and_prepare_data


X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()


print("Training shape:")
print(X_train.shape)


print("\nTesting shape:")
print(X_test.shape)


print("\nFirst training label:")
print(y_train.iloc[0])