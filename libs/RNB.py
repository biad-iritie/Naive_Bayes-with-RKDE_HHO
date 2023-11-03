# Robust Naive Bayes ----------------------------%

#!
# Created by "Boli" at 14:51, 17/08/2023 ----------%
#       Email: biadboze@gmail.com            %
#       Github: https://github.com/biad-iritie        %
# --------------------------------------------------%
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from libs import kde_lib
from libs.exp_lib import Density_model
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm

list_selection = ["hho", "pso"]


class RobustNaiveBayes(BaseEstimator, ClassifierMixin):

    def __init__(self, h_selection="hho") -> None:
        self.class_priors = None
        self.classes = None
        self.kernel = 'gaussian'
        self.classifiers = {}  # Store GaussianNB classifiers for each class
        self.robust_densities = {}  # Store robust densities for each class
        if h_selection in list_selection:
            self.h_selection = h_selection
        else:
            raise ValueError(
                'Should choose a bandwith selection between this list{}'.format(list_selection))

    def fit(self, X, y):
        """
        Fit the robust Naive Bayes model with RKDE densities.

        Parameters:
        X (array-like): Training data features.
        y (array-like): Training data labels.
        """

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate class priors
        self.class_priors = np.array([np.mean(y == c) for c in self.classes])

        for class_label in self.classes:
            # GET for each class
            class_indices = np.where(y == class_label)[0]
            class_data = X[class_indices]

            self.classifiers[class_label] = GaussianNB()
            self.classifiers[class_label].fit(class_data, y[class_indices])

            """ if self.h_selection == "hoo":
                bandwidth = kde_lib.hho_bandwith_selection(
                    class_data, class_data) """
            bandwidth = kde_lib.selection_bandwidth(
                self.h_selection, class_data)
            model = Density_model("rkde", "", 0, self.kernel, bandwidth)
            model.fit(class_data, class_data, grid=None)
            rkde = model.density
            self.robust_densities[class_label] = rkde[:, 0]

    def predict(self, X):
        """
        Predict class labels and RKDE likelihoods for input data.

        Parameters:
        X (array-like): Input data features.

        Returns:
        y_pred (array-like): Predicted class labels.
        rkde_likelihoods (array-like): RKDE likelihoods for each class.
        """
        predictions = []

        for sample in X:
            likelihoods = []

            for class_label in self.classes:
                classifier = self.classifiers[class_label]
                robust_density = self.robust_densities[class_label]
                # Calculate the likelihood using the robust density and GaussianNB classifier
                """ likelihoods.append(np.prod(
                    norm.pdf(sample, loc=robust_density.mean(), scale=robust_density.std()))) """
                log_likelihood = classifier.predict_joint_log_proba(
                    sample.reshape(1, -1))
                likelihood = np.prod(
                    np.exp(log_likelihood * robust_density.reshape(-1, 1)))
                likelihoods.append(likelihood)

            # Normalize likelihoods using class priors
            normalized_likelihoods = likelihoods * self.class_priors

            predicted_class = np.argmax(normalized_likelihoods)
            predictions.append(self.classes[predicted_class])

        return np.array(predictions)
