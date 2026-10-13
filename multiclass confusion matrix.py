import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Choose Database (Hardcoded lists of True values vs Model Predictions)
classes = ['Apple', 'Banana', 'Orange']

actual    = ['Apple', 'Banana', 'Orange', 'Apple', 'Banana', 'Orange', 'Apple', 'Orange', 'Banana', 'Banana']
predicted = ['Apple', 'Orange', 'Orange', 'Apple', 'Banana', 'Orange', 'Banana', 'Orange', 'Banana', 'Apple']

# 2. Compute the Multi-Class Confusion Matrix
matrix = confusion_matrix(actual, predicted, labels=classes)

# 3. Print the Text Analysis Performance Report
print("--- Performance Report ---")
print(classification_report(actual, predicted, target_names=classes))

# 4. Plot the Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted Fruit', fontsize=12)
plt.ylabel('Actual Fruit', fontsize=12)
plt.title('Multi-Class Confusion Matrix Performance', fontsize=14)
plt.show()
