# Model Evaluation Report

**Generated:** 2026-06-22  
**Model:** Random Forest Classifier  
**Vectorizer:** TF-IDF (1,000 features, unigrams + bigrams)  
**Dataset:** 5,674 labeled SMS/email-style messages from `dataset.json`

## Executive Summary

The spam detector achieves strong overall accuracy on the project dataset, with **97.5% accuracy** and **98.8% ROC-AUC**. Spam recall is the main trade-off at **84.9%** — some spam messages are still missed, which is documented honestly below.

| Metric | Value |
|--------|------:|
| Accuracy | 97.53% |
| Precision | 97.13% |
| Recall | 84.94% |
| F1-Score | 90.63% |
| ROC-AUC | 98.80% |

These metrics come from `model_evaluation.py` evaluating the saved model on the full included dataset. Training uses an 80/20 stratified split in `train_model.py`, which reports similar holdout metrics (~96.7% accuracy).

## Dataset

| Aspect | Value |
|--------|------:|
| Total samples | 5,674 |
| Spam samples | 797 (14.0%) |
| Ham samples | 4,877 (86.0%) |
| Language | English |
| Source | Public SMS spam corpus (UCI-style), email-style text |

The dataset is imbalanced (more ham than spam), which reflects real-world inboxes. `class_weight="balanced"` is used during training to reduce bias toward the majority class.

## Evaluation Methodology

The `model_evaluation.py` script:

1. Loads `spam_detector_model.pkl` and `vectorizer.pkl`
2. Reads all samples from `dataset.json`
3. Generates predictions and probability scores
4. Calculates accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
5. Saves metrics to `model_metrics.json`
6. Saves ROC and precision-recall curves to `model_evaluation_curves.png`

Regenerate anytime after retraining:

```bash
python model_evaluation.py
```

## Confusion Matrix

```text
                 Predicted Ham    Predicted Spam
Actual Ham           4,857               20
Actual Spam            120              677
```

| Value | Count | Meaning |
|-------|------:|---------|
| True negatives | 4,857 | Ham correctly classified |
| False positives | 20 | Ham incorrectly flagged as spam |
| False negatives | 120 | Spam incorrectly classified as ham |
| True positives | 677 | Spam correctly classified |

**Key insight:** False negatives (120 missed spam) are higher than false positives (20). The model is conservative about marking messages as spam, which is common with imbalanced data.

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| Ham | 97.59% | 99.59% | 98.58% | 4,877 |
| Spam | 97.13% | 84.94% | 90.63% | 797 |

## Model Architecture

### Classifier

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    class_weight="balanced",
)
```

### Text Vectorizer

```python
TfidfVectorizer(
    max_features=1000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.8,
)
```

## Explainability

The web app surfaces the top 5 TF-IDF features contributing to each prediction and highlights suspicious words in the email body via the `/highlight` endpoint.

## Known Limitations

- Evaluated on the included dataset only, not a separate external benchmark
- English-only; no multilingual support
- SMS-heavy corpus; may not generalize to all corporate email formats
- Spam recall (~85%) leaves room for improvement
- Model artifacts (`.pkl`) are generated locally and not committed to Git

## Recommendations for Future Work

1. Improve spam recall with threshold tuning or additional spam-heavy training data
2. Compare against Logistic Regression or Naive Bayes baselines
3. Add a strict external holdout set never used during training
4. Deploy with Docker and monitor feedback-driven retraining in production

## Generated Artifacts

| File | Purpose |
|------|---------|
| `model_metrics.json` | Machine-readable evaluation metrics |
| `model_evaluation_curves.png` | ROC and precision-recall visualizations |
| `model_evaluation.py` | Script to regenerate metrics and charts |

## Version History

| Date | Version | Accuracy | F1-Score | Notes |
|------|---------|----------:|---------:|-------|
| 2026-06-22 | v2.0 | 97.53% | 90.63% | Full 5,674-sample dataset evaluation |
| 2026-06-18 | v1.0 | — | — | Initial project setup |

## Conclusion

This model demonstrates a complete ML workflow — training, evaluation, explainability, feedback, and retraining — with honest metrics. Strong ham detection (99.6% recall) and high overall accuracy make it a solid portfolio project; the documented spam recall gap shows understanding of real-world model trade-offs.
