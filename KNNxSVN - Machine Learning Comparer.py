# KNN and SVM Code With Confusion Matrices
# Schedule 06 - Artificial Intelligence
# Author: João Pedro Bervalt
# Implemented in Python 3.9
# This code implements a pattern classification application using two machine learning algorithms, K-Nearest Neighbors (KNN) and Support Vector Machine (SVM), to classify data from the Iris dataset. The graphical interface allows the user to choose between the algorithms and perform classification, as well as compare the results between them.
# Loads the Iris dataset, which consists of four attributes (sepal length, sepal width, petal length, and petal width) and three flower classes (setosa, versicolor, and virginica). The graphical interface allows you to select between KNN and SVM algorithms, with KNN configured with 3 neighbors and SVM using a linear kernel.
# Cross-validation is presented, which evaluates the performance of the algorithms. For each algorithm, cross-validation divides the dataset into five parts, training the model on four of them and testing it on the remaining part. This is done for all possible combinations of parts, allowing a robust evaluation of performance.
# After cross-validation, the code calculates confusion matrices for the two algorithms, displaying detailed information about correct and incorrect predictions for each class. This allows a direct comparison between the algorithms in terms of their classification ability.
import tkinter as tk
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class IrisClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Dataset Classification")

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.algo_label = tk.Label(self.frame, text="Select Algorithm:")
        self.algo_label.pack()

        self.algo_var = tk.StringVar(value="KNN")

        self.knn_radio = tk.Radiobutton(self.frame, text="KNN", variable=self.algo_var, value="KNN")
        self.svm_radio = tk.Radiobutton(self.frame, text="SVM", variable=self.algo_var, value="SVM")

        self.knn_radio.pack()
        self.svm_radio.pack()

        self.run_button = tk.Button(self.frame, text="Run Classification", command=self.run_classification)
        self.run_button.pack()

        self.compare_button = tk.Button(self.frame, text="Compare Algorithms", command=self.compare_algorithms)
        self.compare_button.pack()

        self.result_label = tk.Label(self.frame, text="")
        self.result_label.pack()

        # Load the Iris dataset only once at the start of the application.
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target

    def run_classification(self):
        algorithm = self.algo_var.get()
        
        # Clear input data before adding new data.
        self.X = self.iris.data
        self.y = self.iris.target

        if algorithm == "KNN":
            # Hyperparameter Tuning: Try different values for n_neighbors.
            classifier = KNeighborsClassifier(n_neighbors=5)  # Try different values of n_neighbors.
        elif algorithm == "SVM":
            # Hyperparameter Tuning: Experiment with the RBF kernel and adjust the C parameter.
            classifier = SVC(kernel='rbf', C=1.0)  # Try different values of C and kernels.
        else:
            return

        # Test Set Split: Split the dataset into training and test sets.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Data Augmentation: Shuffle and duplicate data to increase the data quantity.
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_train = X_train + X_train
        y_train = y_train + y_train

        # Feature Selection: Use SelectKBest with ANOVA to select the best features.
        feature_selector = SelectKBest(score_func=f_classif, k=3)
        X_train = feature_selector.fit_transform(X_train, y_train)

        # Feature Scaling: Standardize features to have zero mean and unit variance.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        predictions = cross_val_predict(classifier, X_train, y_train, cv=5)

        accuracy = accuracy_score(y_train, predictions)
        precision = precision_score(y_train, predictions, average='weighted')
        recall = recall_score(y_train, predictions, average='weighted')
        f1 = f1_score(y_train, predictions, average='weighted')

        cm = confusion_matrix(y_train, predictions)

        result_text = f"Results for {algorithm}:\n"
        result_text += f"Accuracy: {accuracy:.2f}\n"
        result_text += f"Precision: {precision:.2f}\n"
        result_text += f"Recall: {recall:.2f}\n"
        result_text += f"F1-Score: {f1:.2f}\n"
        result_text += "Confusion Matrix:\n" + "\n".join(["\t".join(map(str, row)) for row in cm])

        self.result_label.config(text=result_text)

    def compare_algorithms(self):
        # Dimensionality Reduction: Reduce dimensionality to evaluate the impact on accuracy.
        feature_selector = SelectKBest(score_func=f_classif, k=2)
        X_reduced = feature_selector.fit_transform(self.X, self.y)

        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        svm_classifier = SVC(kernel='rbf', C=1.0)

        knn_predictions = cross_val_predict(knn_classifier, X_reduced, self.y, cv=5)
        svm_predictions = cross_val_predict(svm_classifier, X_reduced, self.y, cv=5)

        knn_accuracy = accuracy_score(self.y, knn_predictions)
        svm_accuracy = accuracy_score(self.y, svm_predictions)

        knn_cm = confusion_matrix(self.y, knn_predictions)
        svm_cm = confusion_matrix(self.y, svm_predictions)

        knn_cm_str = "KNN Confusion Matrix:\n" + "\n".join(["\t".join(map(str, row)) for row in knn_cm])
        svm_cm_str = "SVM Confusion Matrix:\n" + "\n".join(["\t".join(map(str, row)) for row in svm_cm])

        result_text = "Algorithm Comparison:\n"
        result_text += f"KNN Accuracy: {knn_accuracy:.2f}\n"
        result_text += f"SVM Accuracy: {svm_accuracy:.2f}\n"
        result_text += knn_cm_str + "\n" + svm_cm_str

        self.result_label.config(text=result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = IrisClassifierApp(root)
    root.mainloop()
