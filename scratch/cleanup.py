import json

with open("user_feedback.json", "r") as f:
    data = json.load(f)

# Filter out the Chase phishing emails labeled as ham
filtered_data = [d for d in data if not ('Chase' in d['text'] and d['label'] == 'ham')]

with open("user_feedback.json", "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"Removed {len(data) - len(filtered_data)} bad entries.")
