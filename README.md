# Iris_Classification
This project focuses on building a machine learning model to categorize iris 
flowers into three distinct species: Setosa, Versicolour, and Virginica. 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Iris.csv")

df.head(3)

df.isnull().sum()

# Separate input and output data

inP = df.iloc[:,:-1]
outP = df["Species"]

inP.head(3)

outP.head(3)

# Visualize the dataset

sns.pairplot(df, hue="Species")

plt.show()

# Selecting the first 100 rows from column index 4 (target variable)
y = df.iloc[0:100, 4].values  

# Converting target variable values: If 'Iris-setosa', assign -1; otherwise, assign 1
y = np.where(y == 'Iris-setosa', -1, 1)  

# Selecting the first 100 rows and columns with index 0 and 2 (features)
X = df.iloc[0:100, [0, 2]].values


# Plotting the data
plt.figure(figsize=(7,5))  # Set figure size
plt.scatter(X[:50, 0], X[:50, 1], color='r', marker='*', label='setosa')  # Plot setosa
plt.scatter(X[50:100, 0], X[50:100, 1], color='b', marker='v', label='versicolor')  # Plot versicolor
plt.xlabel('sepal length [cm]')  # X-axis label
plt.ylabel('petal length [cm]')  # Y-axis label
plt.legend(loc='upper left')  # Display legend in upper left corner
plt.show()  # Show the plot

class CustomNeuron:
    """
    A simple neural unit implementing a perceptron algorithm.
    """
    
    def __init__(self, rate=0.01, cycles=50, seed=42):
        """
        Initializes the perceptron with a learning rate, number of iterations, and random seed.
        """
        self.rate = rate
        self.cycles = cycles
        self.seed = seed
        self.weights = None
        self.misclassifications = []
    
    def initialize_weights(self, input_dim):
        """Initializes weights randomly based on the input dimension."""
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0, 0.01, size=input_dim + 1)
    
    def train(self, data, labels):
        """
        Trains the perceptron model on the provided dataset.
        """
        self.initialize_weights(data.shape[1])
        
        for _ in range(self.cycles):
            error_count = 0
            for features, target in zip(data, labels):
                adjustment = self.rate * (target - self.classify(features))
                self.weights[1:] += adjustment * features
                self.weights[0] += adjustment  # Adjust bias term
                error_count += int(adjustment != 0.0)
            self.misclassifications.append(error_count)
        
        return self
    
    def compute_signal(self, data):
        """Computes the weighted sum (net input) for the given data."""
        return np.dot(data, self.weights[1:]) + self.weights[0]
    
    def classify(self, data):
        """Determines the class label based on the net input."""
        return np.where(self.compute_signal(data) >= 0.0, 1, -1)


ppn = CustomNeuron(rate=0.1, cycles=10)

ppn.train(X, y)

plt.plot(range(1, len(ppn.misclassifications) + 1), ppn.misclassifications, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Perceptron Learning Progress')
plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    Plots decision boundaries for a classifier.

    Parameters:
    -----------
    X : ndarray
        Feature set with shape (n_samples, n_features).
    y : ndarray
        Target labels.
    classifier : object
        A trained classifier with a `classify` method.
    resolution : float, optional
        Grid resolution for contour plot (default is 0.02).
    """

    # Define markers and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Determine min and max values for the grid
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Predict class labels for each grid point
    Z = classifier.classify(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Plot decision boundary
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Scatter plot of class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx],
                    label=cl, edgecolor='black')

data=df.drop_duplicates(subset="Species",)

df.value_counts("Species")

plt.figure(figsize=(4,3))

sns.countplot(x='Species', data=df)

plt.show()

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=df,hue='Species' )

plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm',
hue='Species', data=df, )
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "SepalLengthCm").add_legend()
plt.title("Distribution of Sepal Length for Different Species",fontsize=9, fontweight='bold', fontfamily='serif')

plt.savefig("SepalLengthCm.png", dpi=1500,bbox_inches='tight')

plt.show()

plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "SepalWidthCm").add_legend()
plt.title("Distribution of Sepal Width for Different Species",fontsize=9, fontweight='bold', fontfamily='serif')

plt.savefig("SepalWidthCm.png", dpi=1500,bbox_inches='tight')

plt.show()

plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "PetalLengthCm").add_legend()
plt.title("Distribution of Petal Length for Different Species",fontsize=9, fontweight='bold', fontfamily='serif')

plt.savefig("PetalLengthCm.png",dpi=1500,bbox_inches='tight')

plt.show()

plot = sns.FacetGrid(df, hue="Species")
plot.map(sns.distplot, "PetalWidthCm").add_legend()
plt.title("Distribution of Petal Width for Different Species",fontsize=9, fontweight='bold', fontfamily='serif')

plt.savefig("PetalWidthCm.png", dpi=1500,bbox_inches='tight')

plt.show()

def graph(y):
 sns.boxplot(x="Species", y=y, data=df)
 plt.figure(figsize=(10,10))
 # Adding the subplot at the specified
 # grid position
 plt.subplot(221)
 graph('SepalLengthCm')
 plt.subplot(222)
 graph('SepalWidthCm')
 plt.subplot(223)
 graph('PetalLengthCm')
 plt.subplot(224)
 graph('PetalWidthCm')
 plt.show()
 
 sns.boxplot(x='SepalWidthCm', data=df)
 plt.show()

 # IQR
q1 = np.percentile(df['SepalWidthCm'], 25,interpolation = 'midpoint')
q3 = np.percentile(df['SepalWidthCm'], 75,interpolation = 'midpoint')
IQR = q3- q1

print("Old Shape: ", df.shape)

# Upper bound
upper = np.where(df['SepalWidthCm'] >= (q3+1.5*IQR))

# Lower bound
lower = np.where(df['SepalWidthCm'] <= (q1-1.5*IQR))

# Removing the Outliers
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)

print("New Shape: ", df.shape)

sns.boxplot(x='SepalWidthCm', data=df)

plt.show()


# LogisticRegression

# Train the model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score

x_train, x_test, y_train, y_test = train_test_split(inP,outP, test_size=0.2, random_state=43)

lr = LogisticRegression()
lr.fit(x_train,y_train)

lr.score(x_test,y_test)*100

# Examples
lr.predict([[7.2,3.2,6,1.8]])
lr.predict([[5.9,3,5.1,1.8]])
lr.predict([[5.5,2.6,4.4,1.2]])

# Model Performance & Evaluation

y_pred = lr.predict(x_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for LogisticRegression")

plt.savefig("LogisticRegression.jpg", dpi=2000, bbox_inches="tight")

plt.show()

# Precision Score
precision_score(y_test,y_pred, average = 'weighted')*100

# Recall Score
recall_score(y_test,y_pred, average = 'macro')*100

# F1 Score
f1_score(y_test,y_pred, average = 'micro')*100

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
from sklearn.datasets import load_iris

# Initialize Decision Tree Model
# dtc = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=3, random_state=48)

# Train the model
dtc.fit(x_train, y_train)

# Predict on test data
y_pred = dtc.predict(x_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)

# Classification Report
print("Classification Report:\n", (classification_report(y_test, y_pred)))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for DecisionTreeClassifier")

plt.savefig("DecisionTreeClassifier.jpg", dpi=2000, bbox_inches="tight")


plt.show()

# Precision Score
precision_score(y_test,y_pred, average = 'weighted')*100

# Recall Score
recall_score(y_test,y_pred, average = 'macro')*100

# F1 Score
f1_score(y_test,y_pred, average = 'micro')*100

# Plot the Decision Tree
plt.figure(figsize=(7, 6))
tree.plot_tree(dtc, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree")

plt.savefig("DecisionTree.jpg", dpi=2000, bbox_inches="tight")

plt.show()

# SVM(Support Vector Machine) Classifier

from sklearn.svm import SVC

# Initialize SVM model with Radial Basis Function (RBF) kernel
svm_model = SVC(kernel='linear', C=1.0, gamma='scale')

# Train the model
svm_model.fit(x_train, y_train)

# Predict on test data
y_pred = svm_model.predict(x_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100,"%")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Oranges", fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Support Vecotr Machine")

plt.savefig("Support Vecotr Machine.jpg", dpi=2000, bbox_inches="tight")

plt.show()

# Precision Score
precision_score(y_test,y_pred, average = 'weighted')*100

# Recall Score
recall_score(y_test,y_pred, average = 'macro')*100

# F1 Score
f1_score(y_test,y_pred, average = 'micro')*100

# KNN Classifier

from sklearn.neighbors import KNeighborsClassifierKNN Classifier

# Initialize KNN with k=5 (default)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(x_train, y_train)

# Predict on test data
y_pred = knn.predict(x_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Greens", fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix For KNNeighborsClassifier")

plt.savefig("KNNeighborsClassifier.jpg", dpi=2000, bbox_inches="tight")

plt.show()

# Precision Score
precision_score(y_test,y_pred, average = 'weighted')*100

# Recall Score
recall_score(y_test,y_pred, average = 'macro')*100

# F1 Score
f1_score(y_test,y_pred, average = 'micro')*100

# Test different k values
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(f"k={k}, Accuracy={accuracy_score(y_test, y_pred)*100:.4f}")

    
sns.scatterplot(data=df, x=df['PetalLengthCm'], y=df['PetalWidthCm'], hue='Species', palette='cool')

plt.savefig("Distinguishable.jpg", dpi=2000, bbox_inches="tight")

plt.show()


















