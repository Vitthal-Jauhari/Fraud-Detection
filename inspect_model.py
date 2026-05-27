import joblib

model = joblib.load(r"C:\Codes\fraud_detection\Fraud_Detection_Pipeline.pkl")

print("=" * 50)
print("MODEL TYPE")
print("=" * 50)
print(type(model))

print("\n" + "=" * 50)
print("PIPELINE STEPS")
print("=" * 50)
print(model.named_steps)

preprocessor = model.named_steps['prep']

print("\n" + "=" * 50)
print("TRANSFORMERS")
print("=" * 50)
print(preprocessor.transformers_)

encoder = preprocessor.named_transformers_['categorical']

print("\n" + "=" * 50)
print("ENCODER CATEGORIES")
print("=" * 50)
print(encoder.categories_)