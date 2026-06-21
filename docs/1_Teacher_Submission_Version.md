# Email Spam Detector with Self-Learning Feedback
**Final Project Submission Documentation**

## 1. Project Overview
This project is an end-to-end Machine Learning web application designed to classify email text as either *Spam* or *Legitimate (Ham)*. It was built to demonstrate a complete ML lifecycle, from data ingestion and model training to web deployment and continuous self-learning.

## 2. Architecture & Technologies
- **Backend Framework:** Flask (Python)
- **Machine Learning Library:** Scikit-Learn
- **Algorithm:** Random Forest Classifier
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Deployment & CI/CD:** Docker, GitHub Actions
- **Testing:** Pytest (100% Pass Rate across 58 unit tests)
- **Frontend Design:** Premium Dark-Mode UI with Glassmorphism and CSS Micro-animations.

## 3. Dataset & Model Performance
The model is trained on a robust, real-world dataset comprising over **5,000 email samples**. 
- **Total Samples:** 5,764
- **Accuracy:** 97.31%
- **Precision:** 93.87%
- **Recall:** 87.93%
- **F1-Score:** 0.9080

## 4. Key Engineering Features
1. **Explainable AI (XAI):** The application doesn't just predict "Spam"; it exposes the underlying `feature_importances_` array to highlight the exact words (features) that contributed most heavily to the model's decision.
2. **Self-Learning Loop (MLOps):** Users can provide feedback on predictions. If the model misclassifies an email, the user's correction is saved. A dedicated `/retrain` endpoint dynamically re-ingests this feedback, combining it with the base dataset to retrain the model and vectorizer on the fly.
3. **Robust Error Handling:** Comprehensive input validation and structured logging ensure the application handles malformed data gracefully without crashing.
4. **Containerization:** The entire application, including its dependencies, is containerized using Docker, ensuring perfectly reproducible environments across any system.

## 5. Conclusion
This project successfully demonstrates advanced software engineering, applied machine learning, and modern UI/UX capabilities, making it a highly production-ready portfolio asset.
