# 🚀 Spam Detector with Self-Learning Model

A beginner-friendly machine learning project to detect spam emails with real-time model retraining based on user feedback.

## 🎯 What This Project Does

- ✅ **Detects spam emails** using a trained Random Forest classifier
- ✅ **Learns from your feedback** - when model confidence is low, ask for corrections
- ✅ **Retrains itself** with new user data for continuous improvement
- ✅ **Interactive web UI** - paste emails and get instant predictions
- ✅ **Educational** - learn how AI models use your computer's resources
- ✅ **Form validation** - robust error handling for user inputs

## 🛠 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Steps

```bash
# 1. Clone or navigate to the project
cd Spam-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (if not already trained)
python train_model.py

# 4. Run the web application
python app.py
```

The app will automatically open in your browser at `http://127.0.0.1:5000/`

## � Development Setup

To contribute or run tests:

```bash
# 1. Install development dependencies
pip install -r requirements-dev.txt

# 2. Run tests
pytest -v

# 3. Check coverage
pytest --cov=. --cov-report=html
```

**Development Tools:**
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatter
- `flake8` - Linter

```
Spam-detector/
├── app.py                          # Flask web application
├── train_model.py                  # Model training with evaluation metrics
├── check_model.py                  # Quick CLI testing script
├── spam-detector-project.py        # Hardware monitoring demo
├── dataset.json                    # Training dataset (1000+ samples)
├── user_feedback.json              # User corrections (auto-created)
├── spam_detector_model.pkl         # Trained model (binary)
├── vectorizer.pkl                  # TF-IDF vectorizer (binary)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── templates/
    ├── landing.html               # Home page
    ├── index.html                 # Email input form
    └── result.html                # Prediction result & feedback
```

## 🚀 Quick Start

### 1. Train the Model

```bash
python train_model.py
```

**Output:**
- Model accuracy on test set
- Precision, Recall, and F1-Score metrics
- Classification report
- Saves: `spam_detector_model.pkl` and `vectorizer.pkl`

### 2. Run Web Application

```bash
python app.py
```

Then open `http://127.0.0.1:5000/` in your browser

**Features:**
- **Landing page** (`/`) - Introduction and navigation
- **Check page** (`/checkpage`) - Paste email text here
- **Result page** (`/check`) - See spam/real prediction with confidence
- **Feedback** - Correct model if confidence is low

### 3. Test from CLI

```bash
python check_model.py
```

Runs 3 example emails and shows predictions.

## 🧠 How Self-Training Works

### When Confidence is Low

If the model's prediction confidence is **below 60%**, you'll see:

```
⚠️ LOW CONFIDENCE - Help improve the model!
```

With buttons to correct the prediction.

### Feedback Collection

1. User submits email for checking
2. Model makes prediction with confidence score
3. If confidence < 60%, show feedback buttons
4. User corrects if needed → Saved to `user_feedback.json`
5. Model uses all feedback for retraining

### Retraining the Model

**Manual retraining:**
```bash
python train_model.py
```

This automatically:
- Loads original `dataset.json`
- Adds all user corrections from `user_feedback.json`
- Retrains with combined dataset
- Shows updated metrics
- Saves improved model

**Automatic via API:**
```bash
curl -X POST http://127.0.0.1:5000/retrain
```

## 📈 Model Metrics

After training, you'll see:

- **Accuracy** - Percentage of correct predictions
- **Precision** - Of detected spam, how many are real spam
- **Recall** - How many actual spam emails were caught
- **F1-Score** - Balance between precision and recall

Example output:
```
Model Accuracy: 94.23%
Precision: 0.9234
Recall: 0.9156
F1-Score: 0.9195
```

## 🔧 Configuration

Edit `app.py` to adjust:

```python
UNCERTAINTY_THRESHOLD = 0.60  # Show feedback if confidence below this
FEEDBACK_FILE = 'user_feedback.json'  # Where to store corrections
```

## 📚 Learning Resources

This project teaches:

1. **Machine Learning Basics**
   - Text vectorization (TF-IDF)
   - Classification algorithms (Random Forest)
   - Train/test split methodology

2. **Model Evaluation**
   - Accuracy, Precision, Recall
   - Classification reports
   - Confidence scoring

3. **Software Engineering**
   - API design (Flask endpoints)
   - Data validation
   - Error handling
   - Continuous improvement patterns

4. **Hardware Awareness**
   - See `spam-detector-project.py` for CPU/RAM monitoring

## 🚨 Error Handling

The app handles:

- ✅ Empty email text
- ✅ Text too short (< 5 chars)
- ✅ Text too long (> 10,000 chars)
- ✅ Model not loaded
- ✅ Invalid feedback submissions
- ✅ File I/O errors

## 📈 Model Performance

The trained model achieves **excellent results** on evaluation:

- ✅ **Accuracy:** 100%
- ✅ **Precision:** 100%
- ✅ **Recall:** 100%
- ✅ **F1-Score:** 1.0

**📊 For detailed evaluation metrics, confusion matrix, ROC curves, and performance analysis, see [EVALUATION.md](./EVALUATION.md)**

## 🧪 Testing

This project includes a comprehensive test suite with 58 unit tests and 73% code coverage.

### Run All Tests

```bash
# Install dev dependencies first
pip install -r requirements-dev.txt

# Run tests with coverage report
pytest -v --cov=. --cov-report=html

# View coverage report
# Open htmlcov/index.html in your browser
```

### Test Coverage

- **58 Total Tests** - 100% pass rate
- **73% Code Coverage** - Exceeds professional standards
- **Test Categories:**
  - Model tests (16 tests) - Loading, prediction, metrics
  - Flask app tests (24 tests) - Routes, endpoints, error handling
  - Validation tests (18 tests) - Edge cases and boundary conditions

### Test Results

```
tests/test_model.py         - 95% coverage (16 tests)
tests/test_validation.py    - 93% coverage (18 tests)
tests/conftest.py          - 83% coverage (fixtures)
app.py                     - 64% coverage (24 tests)
```

**Execution time:** 1.04 seconds

## 📊 Project Quality Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Code Coverage | 73% | ✅ Excellent |
| Test Count | 58 | ✅ Comprehensive |
| Test Pass Rate | 100% | ✅ All passing |
| Lines of Test Code | 648 | ✅ Professional |

## 📱 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Landing page |
| `/checkpage` | GET | Email input form |
| `/check` | POST | Submit email for classification |
| `/feedback` | POST | Submit user correction |
| `/retrain` | POST | Retrain model with feedback |
| `/status` | GET | Get model status & stats |

## 🎓 Example Usage

### Via Web UI

1. Go to `http://127.0.0.1:5000/`
2. Click "Check Email"
3. Paste email text
4. Click "Check Email"
5. See result: SPAM or REAL
6. If confidence is low, click "Actually [SPAM/REAL]" to correct
7. Model learns and improves

### Via Python

```python
import pickle

# Load model
model = pickle.load(open('spam_detector_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Test email
email = "Congratulations! You won $1,000,000!"
X = vectorizer.transform([email])
prediction = model.predict(X)[0]
confidence = model.predict_proba(X)[0]

print(f"SPAM: {prediction == 1}")  # Output: SPAM: True
print(f"Confidence: {max(confidence)*100:.1f}%")  # Output: Confidence: 95.2%
```

## 🔄 Workflow

```
1. User trains model (train_model.py)
   ↓
2. Runs web app (app.py)
   ↓
3. Tests emails via web UI
   ↓
4. Model shows uncertainty? → User provides feedback
   ↓
5. Feedback saved to user_feedback.json
   ↓
6. User retrains model (includes feedback)
   ↓
7. Model improves → repeat from step 3
```

## 📊 Dataset

- **Source**: `dataset.json` (1000+ labeled emails)
- **Labels**: "spam" or "ham" (real email)
- **Split**: 80% training, 20% testing
- **Features**: TF-IDF vectorized text (1000 features)

## ⚡ Performance

- **Model Training**: ~2-5 seconds
- **Single Prediction**: <50ms
- **Web Page Load**: <100ms
- **Model File Size**: ~2-3 MB

## 🐛 Troubleshooting

### Model file not found
```
Error: Model files not found. Please run train_model.py first.
```
**Solution**: `python train_model.py`

### Port already in use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Kill the process or use different port in `app.py`:
```python
app.run(port=5001)  # Use 5001 instead of 5000
```

### Import errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: `pip install -r requirements.txt`

## 👤 Author

**Bibiayesha Badeghar**
- Email: ayeshabadeghar2@gmail.com
- GitHub: [@Bibiayesha-Badeghar](https://github.com/Bibiayesha-Badeghar)

## 📝 License

This project is open source and available for educational purposes.

## ✨ Key Features Summary

| Feature | Status |
|---------|--------|
| Web UI | ✅ |
| Model Training | ✅ |
| Predictions | ✅ |
| Confidence Scoring | ✅ |
| Form Validation | ✅ |
| Error Handling | ✅ |
| Self-Learning | ✅ |
| Feedback Collection | ✅ |
| Model Retraining | ✅ |
| Evaluation Metrics | ✅ |
| API Endpoints | ✅ |

---

**Made with ❤️ learning about AI and GitHub**

