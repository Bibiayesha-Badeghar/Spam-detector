"""
SPAM DETECTOR PROJECT
=====================
Goal: Build a tiny spam detector to see how software (AI model) 
runs in hardware (CPU, RAM, disk).

What you'll learn:
1. How to train a machine learning model
2. How to save and load models
3. How to monitor CPU & RAM usage
4. How AI models use your computer's resources
"""

# Step 1: Install required libraries first
# Run this in terminal: pip install scikit-learn psutil

import psutil  # Library to monitor CPU, RAM, Disk
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

print("=" * 60)
print("STEP 1: PREPARE TRAINING DATA")
print("=" * 60)

# Sample data: Spam and Real emails
training_data = [
    # SPAM EMAILS
    ("Free money! Click here NOW!!!", 1),
    ("Buy cheap products at unbeatable prices", 1),
    ("Congratulations! You won a lottery!", 1),
    ("Act now or lose this offer forever", 1),
    ("LIMITED TIME: Get 90% discount", 1),
    
    # REAL EMAILS
    ("Hi, let's schedule a meeting tomorrow", 0),
    ("Can you review my code changes?", 0),
    ("The project deadline is next Friday", 0),
    ("Thanks for the meeting notes", 0),
    ("Please update the documentation", 0),
]

# Separate emails and labels
emails = [item[0] for item in training_data]
labels = [item[1] for item in training_data]

print(f"Total emails: {len(emails)}")
print(f"Total spam: {sum(labels)}")
print(f"Total real: {len(labels) - sum(labels)}")
print()

# ============================================================
print("=" * 60)
print("STEP 2: CHECK HARDWARE BEFORE TRAINING")
print("=" * 60)

def show_hardware_stats(stage):
    """Show current CPU, RAM, and Disk usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"\n📊 {stage}:")
    print(f"   CPU Usage: {cpu_percent}%")
    print(f"   RAM Used: {ram.used / (1024**3):.2f} GB / {ram.total / (1024**3):.2f} GB ({ram.percent}%)")
    print(f"   Disk Used: {disk.used / (1024**3):.2f} GB / {disk.total / (1024**3):.2f} GB ({disk.percent}%)")

show_hardware_stats("🔍 Before training")

# ============================================================
print("\n" + "=" * 60)
print("STEP 3: CONVERT TEXT TO NUMBERS (So AI can understand)")
print("=" * 60)

vectorizer = CountVectorizer()
email_numbers = vectorizer.fit_transform(emails)

print(f"\nConverted {len(emails)} emails to a table of numbers")
print(f"Table size: {email_numbers.shape}")
print(f"(This means: {email_numbers.shape[0]} emails, {email_numbers.shape[1]} unique words)")

# ============================================================
print("\n" + "=" * 60)
print("STEP 4: TRAIN THE MODEL (This uses CPU)")
print("=" * 60)

show_hardware_stats("⚡ Just before training")

model = MultinomialNB()
model.fit(email_numbers, labels)

show_hardware_stats("📊 Just after training")

print("\n✅ Model trained successfully!")
print(f"Model learned to classify emails with this accuracy:")
print(f"   Training accuracy: {model.score(email_numbers, labels) * 100:.1f}%")

# ============================================================
print("\n" + "=" * 60)
print("STEP 5: SAVE MODEL TO DISK")
print("=" * 60)

model_file = "spam_detector_model.pkl"
vectorizer_file = "vectorizer.pkl"

with open(model_file, "wb") as f:
    pickle.dump(model, f)
    
with open(vectorizer_file, "wb") as f:
    pickle.dump(vectorizer, f)

model_size = os.path.getsize(model_file) / 1024
print(f"\n💾 Model saved to disk!")
print(f"   File: {model_file}")
print(f"   Size: {model_size:.2f} KB")

show_hardware_stats("📊 After saving (RAM is freed)")

# ============================================================
print("\n" + "=" * 60)
print("STEP 6: LOAD MODEL FROM DISK (For future use)")
print("=" * 60)

del model, vectorizer, email_numbers

print("🗑️  Deleted model from RAM (simulating computer restart)")
show_hardware_stats("📊 After deletion")

print("\n⏳ Loading model from disk...")
with open(model_file, "rb") as f:
    loaded_model = pickle.load(f)
    
with open(vectorizer_file, "rb") as f:
    loaded_vectorizer = pickle.load(f)

print("✅ Model loaded from disk into RAM")
show_hardware_stats("📊 After loading")

# ============================================================
print("\n" + "=" * 60)
print("STEP 7: MAKE PREDICTIONS (Test the model)")
print("=" * 60)

test_emails = [
    "Free money! Click here!",
    "Can you review this code?",
    "Congratulations you won!",
    "Meeting at 3pm tomorrow",
]

print(f"\nTesting {len(test_emails)} new emails:\n")

show_hardware_stats("⚡ Before making predictions")

for test_email in test_emails:
    test_numbers = loaded_vectorizer.transform([test_email])
    prediction = loaded_model.predict(test_numbers)[0]
    probability = loaded_model.predict_proba(test_numbers)[0]
    
    emoji = "🚫" if prediction == 1 else "✅"
    spam_or_real = "SPAM" if prediction == 1 else "REAL"
    confidence = max(probability) * 100
    
    print(f"{emoji} '{test_email}'")
    print(f"   → {spam_or_real} (Confidence: {confidence:.1f}%)\n")

show_hardware_stats("📊 After making predictions")

# ============================================================
print("\n" + "=" * 60)
print("STEP 8: KEY OBSERVATIONS")
print("=" * 60)

print("""
✅ What you just saw:

1. TRAINING (Step 4)
   • CPU usage spiked 📈
   • RAM filled with data and model
   • This is where the learning happened

2. SAVING (Step 5)
   • Model written to disk as a .pkl file
   • RAM is now free to use for other things
   • This is like taking a photo of a trained brain

3. LOADING (Step 6)
   • Model loaded back from disk into RAM
   • Took less time than training!
   • This is like waking up a sleeping brain

4. PREDICTING (Step 7)
   • CPU usage was LOW ✅
   • Model ran FAST
   • This is why AI is useful - once trained, predictions are cheap!
""")

# Clean up
os.remove(model_file)
os.remove(vectorizer_file)
print("🧹 Cleaned up: Deleted model files")