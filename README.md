# Amazon Food Review Sentiment Analysis Using Machine Learning

A comparative study of six classical machine learning classifiers for binary sentiment classification (Positive / Negative) of **100,000 Amazon Fine Food reviews**, using a bag-of-words pipeline in Python (scikit-learn, NLTK) with an equivalent R implementation.

![Pipeline](figures/fig1_pipeline.png)

## Highlights

- **Best accuracy:** Support Vector Classifier, **88.29%** (F1 = 0.927)
- **Best balanced accuracy:** Multinomial Naive Bayes, **0.802**
- Accuracy alone is misleading on this imbalanced data (77% positive): tree ensembles score high accuracy but miss most negative reviews.

## Dataset

| Property | Value |
|---|---|
| File | `amazon Food Reviews 100k Dataset.csv` (inside `archive (1).zip`) |
| Rows | 100,000 |
| Columns | `Id`, `Rating` (1–5), `Review` (text) |
| Missing values | 0 |
| Median review length | 57 words |

**Labelling rule:** `Rating > 3` → Positive, `Rating ≤ 3` → Negative, giving 77,055 Positive (77.1%) and 22,945 Negative (22.9%) reviews.

| Rating | Count | Share | Label |
|---|---|---|---|
| 1 | 9,318 | 9.3% | Negative |
| 2 | 5,568 | 5.6% | Negative |
| 3 | 8,059 | 8.1% | Negative |
| 4 | 14,643 | 14.6% | Positive |
| 5 | 62,412 | 62.4% | Positive |

![Ratings](figures/fig2_ratings.png)
![Classes](figures/fig3_classes.png)

## Method

1. **Pre-processing:** lower-case, remove non-letter characters, tokenise (NLTK), remove English stop words.
2. **Split:** 80% train / 20% test, `random_state=42`.
3. **Features:** `CountVectorizer(max_features=5000)`, fitted on training data only.
4. **Classifiers (scikit-learn defaults):** Multinomial Naive Bayes, Logistic Regression, SVC (RBF), Decision Tree, Random Forest, Gradient Boosting.
5. **Metrics:** accuracy, precision, recall, F1 (Positive class), confusion matrix; plus negative-class recall, balanced accuracy and macro-F1 derived from the confusion matrices.

## Results

| Classifier | Accuracy | Precision | Recall | F1 | Neg. recall | Balanced acc. | Macro-F1 |
|---|---|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.8576 | **0.9101** | 0.9044 | 0.9072 | **0.700** | **0.802** | 0.800 |
| Logistic Regression | 0.8750 | 0.9009 | 0.9413 | 0.9207 | 0.653 | 0.797 | **0.813** |
| **Support Vector Classifier** | **0.8829** | 0.8888 | 0.9693 | **0.9273** | 0.593 | 0.781 | **0.813** |
| Decision Tree | 0.8152 | 0.8782 | 0.8824 | 0.8803 | 0.590 | 0.736 | 0.737 |
| Random Forest | 0.8681 | 0.8656 | 0.9812 | 0.9198 | 0.489 | 0.735 | 0.775 |
| Gradient Boosting | 0.8119 | 0.8114 | **0.9847** | 0.8897 | 0.232 | 0.609 | 0.626 |

![Metrics](figures/fig6_metrics.png)
![Confusion matrices](figures/fig7_confusion.png)
![Balanced metrics](figures/fig9_balanced.png)

## Repository structure

```
├── CODE_py.ipynb        # Colab notebook with outputs
├── code_py (1).py       # Python script exported from the notebook
├── PYTHON CODE          # Python script (same pipeline)
├── R CODE               # Equivalent R implementation (tm, caret, e1071, randomForest, gbm)
└── archive (1).zip      # Dataset (100k reviews CSV)
```

## How to run (Python)

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
unzip "archive (1).zip"
```

Update `data_path` in the script to point to the extracted CSV, then run:

```bash
python "code_py (1).py"
```

Note: the RBF-kernel SVC is the slowest model and may take a long time on 80,000 training reviews.

## Limitations and future work

- Three-star (neutral) reviews are labelled Negative, adding label noise.
- Stop-word removal deletes negations (*not*, *no*), so “not good” becomes “good”.
- HTML `<br />` tags are not stripped and become the token `br`.
- Default hyper-parameters and a single split; Logistic Regression hits its iteration limit.
- No class-imbalance handling (class weights, threshold tuning, SMOTE).
- Next steps: TF-IDF and n-grams, word embeddings, fine-tuned transformers (e.g. BERT).

## References

- J. McAuley and J. Leskovec, “From amateurs to connoisseurs: Modeling the evolution of user expertise through online reviews,” WWW 2013.
- B. Pang, L. Lee, S. Vaithyanathan, “Thumbs up? Sentiment classification using machine learning techniques,” EMNLP 2002.
- F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” JMLR, 2011.
