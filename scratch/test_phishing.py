import pickle

# Load model and vectorizer
with open('spam_detector_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

email_text = """Subject: URGENT: Suspicious activity on your Chase account

Hello,

We have detected unauthorized login attempts on your Chase bank account.
For your security, please verify your account information immediately
by clicking this link:

http://chase-security-verify.ru/login

If this was not you, change your password immediately.

Chase Security Team"""

# Predict
features = vectorizer.transform([email_text])
prediction = model.predict(features)[0]
confidence = model.predict_proba(features)[0]
classes = model.classes_

print(f"Prediction: {prediction}")
print(f"Confidence: {dict(zip(classes, confidence))}")

# Feature importance analysis
feature_names = vectorizer.get_feature_names_out()
word_importances = []

# Get non-zero features for this email
feature_indices = features.nonzero()[1]
for idx in feature_indices:
    word = feature_names[idx]
    importance = model.feature_importances_[idx]
    word_importances.append({'word': word, 'importance': importance})

# Sort by importance
word_importances.sort(key=lambda x: x['importance'], reverse=True)

print("\nTop 10 most important features in this email:")
for i, item in enumerate(word_importances[:10]):
    print(
        f"{i+1}. {item['word']} "
        f"(importance: {item['importance']:.6f})"
    )
