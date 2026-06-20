"""
Model Evaluation Script
=======================
Comprehensive evaluation of the spam detection model including:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curves
- Detailed Classification Report
- Performance Metrics
"""

import json
import pickle
import logging
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_model_and_vectorizer():
    """Load trained model and vectorizer."""
    try:
        with open("spam_detector_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError as e:
        logger.error(f"Error: Model files not found. {e}")
        return None, None
    except Exception:
        logger.exception("Failed to load model files.")
        return None, None


def evaluate_model():
    """Perform comprehensive model evaluation."""

    logger.info("=" * 70)
    logger.info("SPAM DETECTOR - MODEL EVALUATION")
    logger.info("=" * 70)

    # Load model
    logger.info("[1/6] Loading model and data...")
    model, vectorizer = load_model_and_vectorizer()
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return None

    # Load dataset
    with open("dataset.json", "r") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [1 if item["label"] == "spam" else 0 for item in data]

    # Vectorize
    logger.info("[2/6] Vectorizing text data...")
    X = vectorizer.transform(texts)

    # Predictions
    logger.info("[3/6] Generating predictions...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    logger.info("[4/6] Calculating metrics...")

    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred, zero_division=0)
    recall = recall_score(labels, y_pred, zero_division=0)
    f1 = f1_score(labels, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, y_pred_proba)
    except Exception:
        roc_auc = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()

    # Classification report
    class_report = classification_report(
        labels, y_pred, target_names=["Ham", "Spam"], output_dict=True
    )

    # Generate curves
    logger.info("[5/6] Generating ROC and Precision-Recall curves...")

    fpr, tpr, _ = roc_curve(labels, y_pred_proba)
    roc_auc_curve = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(labels, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)

    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    axes[0].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC Curve (AUC = {roc_auc_curve:.3f})",
    )
    axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Precision-Recall Curve
    axes[1].plot(
        recall_curve,
        precision_curve,
        color="darkgreen",
        lw=2,
        label=f"PR Curve (AUC = {pr_auc:.3f})",
    )
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_evaluation_curves.png", dpi=300, bbox_inches="tight")
    logger.info("   Saved: model_evaluation_curves.png")

    # Save metrics
    logger.info("[6/6] Saving evaluation report...")

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "dataset_info": {
            "total_samples": len(labels),
            "spam_count": int(sum(labels)),
            "ham_count": len(labels) - int(sum(labels)),
            "spam_ratio": f"{sum(labels)/len(labels)*100:.1f}%",
        },
        "performance_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "detailed_metrics": {
            "ham": {
                "precision": float(class_report["Ham"]["precision"]),
                "recall": float(class_report["Ham"]["recall"]),
                "f1_score": float(class_report["Ham"]["f1-score"]),
                "support": int(class_report["Ham"]["support"]),
            },
            "spam": {
                "precision": float(class_report["Spam"]["precision"]),
                "recall": float(class_report["Spam"]["recall"]),
                "f1_score": float(class_report["Spam"]["f1-score"]),
                "support": int(class_report["Spam"]["support"]),
            },
        },
    }

    # Save JSON metrics
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("   Saved: model_metrics.json")

    # Print report
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Dataset: {metrics['dataset_info']['total_samples']} samples")
    logger.info(
        f"  - Spam: {metrics['dataset_info']['spam_count']} ({metrics['dataset_info']['spam_ratio']})"
    )
    logger.info(f"  - Ham:  {metrics['dataset_info']['ham_count']}")

    logger.info("Performance Metrics:")
    logger.info(f"  - Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  - Precision: {precision:.4f}")
    logger.info(f"  - Recall:    {recall:.4f}")
    logger.info(f"  - F1-Score:  {f1:.4f}")
    logger.info(f"  - ROC AUC:   {roc_auc:.4f}")

    logger.info("Confusion Matrix:")
    logger.info(f"  - True Negatives:  {tn}")
    logger.info(f"  - False Positives: {fp}")
    logger.info(f"  - False Negatives: {fn}")
    logger.info(f"  - True Positives:  {tp}")

    logger.info(
        "Detailed Classification Report:\n%s",
        classification_report(labels, y_pred, target_names=["Ham", "Spam"]),
    )

    logger.info("=" * 70)
    logger.info("✅ Evaluation complete!")
    logger.info("=" * 70)

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
