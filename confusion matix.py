import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#2. The Data (What Happened vs. What Was Guessed)
actual = np.array( ['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog'])
predicted = np.array( ['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog'])

#3. The Math (Calculating the Matrix)
conf_matrix = confusion_matrix(actual, predicted)

#4. The Visuals (Drawing the Heatmap)
sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=['Dog','Not Dog'], yticklabels=['Dog','Not Dog'], cmap='RdPu')

#5. Labels and Showing the Plot
plt.ylabel("prediction", fontsize=16)
plt.xlabel("actual", fontsize=16)
plt.title("confusion matrix", fontsize=20)
plt.show()

