
## Exercise Summary
In this exercise, we explored the performance of three classifiers: Logistic Regression, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP). The evaluation was conducted using 5-fold cross-validation, where the dataset was shuffled and split into five train-test subsets using the KFold method.

### Key Steps:
1. Data Preparation: The dataset was shuffled, and the KFold method was employed to create five distinct train-test splits.

2. Feature Extraction: We utilized the TF-IDF (Term Frequency-Inverse Document Frequency) representation of the text descriptions for feature extraction. For each fold, a separate TF-IDF vectorizer was created based on the training data.

3. Model Training and Testing: Each classifier was trained on the training data of the respective fold and evaluated on the test data.

4. Performance Metrics: The average confusion matrix across the five folds was computed, along with the mean values of accuracy, precision, recall, and F1-measure for each class.

### Logistic Regression Insights: 
For the Logistic Regression classifier in the final fold, we identified the 20 words with the highest positive weights and the 20 words with the lowest negative weights. An analysis of these significant words provided insights into their relevance for classification..

### Results and Discussion:
The results were analyzed, highlighting the strengths and weaknesses of each classifier. The performance metrics indicated that the SVM classifier performed best on the dataset, achieving better scores compared to the other classifiers.

### Implementation :
Originaly it was created with Notebook Jupyter and it was also extracted in a Python file. 
