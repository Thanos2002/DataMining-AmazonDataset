import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sklearn.feature_extraction.text as sk_text

df = pd.read_csv("AmazonData.csv")
print(df.head())




logistic_regression = LogisticRegression(solver='lbfgs')
svm_classifier = SVC()
mlp_classifier = MLPClassifier(solver='lbfgs',max_iter = 1000)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

vectorizer = sk_text.TfidfVectorizer(min_df=1)
df['description'] = df['description'].fillna('')
X_tfidf = vectorizer.fit_transform(df['description'])

fold_num = 0
# Dictionaries for organizing the metrics of each model
metrics = {
    'LogisticRegression': {'confusion_matrices': [],'accuracies': [],'precisions': [],'recalls': [],'f1_scores': []},
    'SVC': {'confusion_matrices': [],'accuracies': [],'precisions': [],'recalls': [],'f1_scores': []},
    'MLPClassifier': {'confusion_matrices': [],'accuracies': [],'precisions': [],'recalls': [],'f1_scores': []}
}

for train_index, test_index in kf.split(X_tfidf):
    fold_num+=1
    X_train, X_test =  df['description'][train_index], df['description'][test_index]
    y_train, y_test = df['category'][train_index], df['category'][test_index]

    vectorizer = sk_text.TfidfVectorizer(min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    predictors = [logistic_regression,svm_classifier,mlp_classifier]
    print("Fold:",fold_num)
    for model in predictors:
        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_test_tfidf)

        model_name = type(model).__name__
        print(model_name,model.score(X_test_tfidf,y_test))

        # Store metrics for each model
        metrics[model_name]['confusion_matrices'].append(confusion_matrix(y_test, predictions))
        metrics[model_name]['accuracies'].append(accuracy_score(y_test, predictions))
        metrics[model_name]['precisions'].append(precision_score(y_test, predictions, average=None))
        metrics[model_name]['recalls'].append(recall_score(y_test, predictions, average=None))
        metrics[model_name]['f1_scores'].append(f1_score(y_test, predictions, average=None))

for model_name, model_metrics in metrics.items():
    print(f"Results for {model_name}:")
    print("Confusion Matrices:", np.mean(model_metrics['confusion_matrices'], axis=0))
    print("Accuracies:", np.mean(model_metrics['accuracies']))
    print("Precisions:", np.mean(model_metrics['precisions'], axis=0))
    print("Recalls:", np.mean(model_metrics['recalls'], axis=0))
    print("F1 Scores:", np.mean(model_metrics['f1_scores'], axis=0))
    print("\n")

coefficients = logistic_regression.coef_[0]
pos = coefficients.argsort()[-20:][::-1]
top_pos = vectorizer.get_feature_names_out()[pos]

neg = coefficients.argsort()[:20]
top_neg = vectorizer.get_feature_names_out()[neg]

print("20 words with biggest positive weight:\n", top_pos)
print("20 words with smallest negative weight:\n", top_neg)
