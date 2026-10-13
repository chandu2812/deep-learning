import pandas as pd
import matplotlib.pyplot as plt
import seaborn   as sns # You can also use: import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Hardcoded Data (No external file needed)
# Imagine 10 wines with: Alcohol %, Color Intensity, and Class (0 = Red, 1 = White)
raw_data = {
    'alcohol': [13.2, 14.1, 12.0, 11.5, 13.8, 12.3, 14.0, 11.8, 13.5, 12.1],
    'color':   [ 5.5,  6.2,  2.1,  1.8,  5.1,  2.8,  5.9,  2.3,  4.8,  2.5],
    'Target':  [   0,     0,     1,     1,     0,     1,     0,     1,     0,     1]
}
data = pd.DataFrame(raw_data)

# 2. Split features and labels
x = data.drop('Target', axis=1)
y = data['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# 3. Train the model
model = DecisionTreeClassifier(random_state=1)
model.fit(x_train, y_train)

# 4. Make predictions
y_pred = model.predict(x_test)

# 5. Print text metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Red Wine', 'White Wine']))

# 6. Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
import seaborn as sns # ensures seaborn is loaded locally if needed
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='PuBuGn', 
            xticklabels=['Red Wine', 'White Wine'], 
            yticklabels=['Red Wine', 'White Wine'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
