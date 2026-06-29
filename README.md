# Email Spam Detector with Self-Learning Feedback

[![CI](https://github.com/Bibiayesha-Badeghar/Spam-detector/actions/workflows/tests.yml/badge.svg)](https://github.com/Bibiayesha-Badeghar/Spam-detector/actions/workflows/tests.yml)

A Flask-based machine learning web app that specifically classifies email text as spam or legitimate using TF-IDF features and a Random Forest classifier.

The project is designed as a portfolio-friendly demonstration of a complete ML workflow: data loading, model training, prediction, confidence scoring, user feedback collection, automated tests, and evaluation reporting.

## Features

- Spam vs. legitimate message classification
- Confidence score and explainable prediction output (top contributing words) for each prediction
- Flask web interface for checking messages
- User feedback endpoint for correcting low-confidence predictions
- Retraining workflow using feedback data
- Input validation and error handling
- Automated test suite running on GitHub Actions CI/CD
- Model evaluation report with metrics and visualizations
- Fully configurable via environment variables
- Containerized for easy deployment with Docker
- Structured logging and robust exception handling

## Screenshots

**Landing Page**

![Landing Page](screenshots/landing.png)

**Spam Detection Result**

![Result Page](screenshots/result.png)

## Project Structure

```text
Spam-detector/
├── .github/
│   └── workflows/
│       └── tests.yml               # GitHub Actions CI pipeline
├── app.py                          # Flask web application
├── train_model.py                  # Model training script
├── model_evaluation.py             # Evaluation script for saved model
├── dataset.json                    # 5000+ real-world email samples
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # One-command container setup
├── .dockerignore                   # Docker build context exclusions
├── EVALUATION.md                   # Model evaluation report
├── model_metrics.json              # Generated evaluation metrics
├── model_evaluation_curves.png     # ROC and precision-recall curves
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Development and testing dependencies
├── pytest.ini                      # Pytest configuration
├── user_feedback.example.json      # Safe example feedback file
├── screenshots/
│   ├── landing.png                 # Landing page screenshot
│   └── result.png                  # Result page screenshot
├── templates/
│   ├── landing.html                # Home page
│   ├── index.html                  # Message input form
│   └── result.html                 # Prediction result page
└── tests/
    ├── test_app.py                 # Flask route and endpoint tests
    ├── test_model.py               # Model and dataset tests
    └── test_validation.py          # Input validation tests
```

Generated local files such as `spam_detector_model.pkl`, `vectorizer.pkl`, `user_feedback.json`, `.coverage`, `.pytest_cache/`, and `htmlcov/` are intentionally ignored by Git.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
cd Spam-detector
pip install -r requirements.txt
python train_model.py
python app.py
```

The app runs at:

```text
http://127.0.0.1:5000/
```

Training creates the local model artifacts:

```text
spam_detector_model.pkl
vectorizer.pkl
```

These files are generated from the dataset and are not committed to the repository.

## Development Setup

```bash
pip install -r requirements-dev.txt
pytest -v
pytest --cov=. --cov-report=html
```

Development tools:

- `pytest` for tests
- `pytest-cov` for coverage reporting
- `black` for formatting
- `flake8` for linting
- `matplotlib` and `numpy` for evaluation visualizations

## Docker

Run the entire app with a single command — no local Python installation required:

```bash
docker compose up --build
```

The app will be available at:

```text
http://localhost:5000/
```

To stop the container:

```bash
docker compose down
```

## Usage

### Web App

1. Run `python app.py`
2. Open `http://127.0.0.1:5000/`
3. Click "Start Checking"
4. Paste an email or suspicious message
5. Submit and review the prediction, confidence score, and result details

### Python Example

```python
import pickle

model = pickle.load(open("spam_detector_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

email = "Congratulations! You won $1,000,000!"
X = vectorizer.transform([email])
prediction = model.predict(X)[0]
confidence = model.predict_proba(X)[0]

print(f"SPAM: {prediction == 1}")
print(f"Confidence: {max(confidence) * 100:.1f}%")
```

## Self-Learning Feedback Flow

The app uses a default confidence threshold of `0.60` (configurable via `UNCERTAINTY_THRESHOLD`).

When prediction confidence is low, the result page can ask the user for feedback. Feedback is saved locally to `user_feedback.json`, and `train_model.py` can include that feedback during retraining.

Workflow:

```text
1. User checks a message
2. Model predicts spam or legitimate
3. App shows confidence score
4. Low-confidence predictions can collect user feedback
5. Feedback is saved locally
6. Model can be retrained with original data + feedback
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML | scikit-learn (Random Forest, TF-IDF) |
| Backend | Python, Flask |
| Frontend | HTML, CSS (responsive UI) |
| Testing | pytest, pytest-cov |
| CI/CD | GitHub Actions |
| Deployment | Docker, docker-compose |

## Dataset

The included dataset contains **5,674 labeled messages** (SMS/email-style):

- 797 spam (14.0%)
- 4,877 ham / legitimate (86.0%)
- English-only examples
- Imbalanced, reflecting real-world inboxes

## Model Evaluation

Latest evaluation on the full dataset (see [EVALUATION.md](./EVALUATION.md)):

| Metric | Value |
|--------|------:|
| Accuracy | 97.53% |
| Precision | 97.13% |
| Recall | 84.94% |
| F1-Score | 90.63% |
| ROC-AUC | 98.80% |

Spam recall (~85%) is the main area for improvement — the model misses some spam to avoid false positives. Metrics are regenerated with `python model_evaluation.py` after retraining.

## Testing

The project includes an automated test suite covering routes, model loading, validation, and API security.

Latest local test result:

```text
67 passed
Total coverage: ~74%
```

Test areas:

- Flask app routes and endpoints
- Model and vectorizer loading
- Dataset integrity
- Prediction output shape and confidence ranges
- Input validation boundaries and edge cases

Run tests:

```bash
pytest --quiet
```

Run tests with coverage:

```bash
pytest --cov=. --cov-report=html
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Landing page |
| `/checkpage` | GET | Message input form |
| `/check` | POST | Submit text for spam classification |
| `/feedback` | POST | Submit user correction |
| `/highlight` | POST | Highlight top spam indicators (JSON) |
| `/retrain` | POST | Retrain model with feedback (API key optional) |
| `/status` | GET | Return model and feedback status |

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `false` | Enable debug mode and auto-open browser |
| `UNCERTAINTY_THRESHOLD` | `0.60` | Confidence threshold for feedback prompt |
| `FEEDBACK_FILE` | `user_feedback.json` | Path to feedback storage file |
| `RETRAIN_API_KEY` | *(empty)* | If set, `/retrain` requires `X-API-Key` header |

Example:

```bash
UNCERTAINTY_THRESHOLD=0.75 PORT=8080 python app.py
```

## Error Handling

The app handles:

- Empty input
- Very short input
- Input longer than 10,000 characters
- Missing model files
- Invalid feedback labels
- Feedback file read/write errors

## Limitations

- English-only; SMS-heavy corpus may not cover all email formats
- Spam recall (~85%) — some spam messages are missed
- Model files (`.pkl`) are generated locally and not committed to Git
- Set `RETRAIN_API_KEY` before exposing `/retrain` in production

## Skills Demonstrated

- End-to-end ML pipeline: data loading, vectorization, training, evaluation
- REST API design with input validation and error handling
- Explainable AI: top feature importance and word highlighting
- Human-in-the-loop learning via user feedback and retraining
- Automated testing and CI/CD with GitHub Actions
- Containerization with Docker

## Future Improvements

1. Improve spam recall with threshold tuning or additional training data
2. Deploy to a cloud platform (Render, Railway, or AWS)
3. Add model comparison baselines (Naive Bayes, Logistic Regression)

## License

This project is open source and available for educational purposes.

## Author

**Bibiayesha Badeghar**

- GitHub: [@Bibiayesha-Badeghar](https://github.com/Bibiayesha-Badeghar)

---

Built as a practical portfolio project for learning machine learning, Flask, testing, and production-minded engineering.
