import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        super().__init__()
        self.prob_Y_1 = 0
        self.word_probs_pos = None
        self.word_probs_neg = None
        if (not self):
            raise Exception("Failed to create Naive Bayes Classifier Object")
        

    def fit(self, X, Y):
        # get P(Y = +1)
        self.prob_Y_1 = Y.sum() / len(Y)
        small_val = .1

        # set pos and neg word prob vectors to length of vectors in X
        self.word_probs_pos = np.full(len(X[0]), small_val)
        self.word_probs_neg = np.full(len(X[0]), small_val)

        #add up all the vectors given pos and neg class
        for i in range(len(X)):
            if Y[i] == 1:
                self.word_probs_pos = self.word_probs_pos + X[i]
            else:
                self.word_probs_neg = self.word_probs_neg + X[i]
        
        # divide each word count by the total number of words given pos class
        # gives prob of word given pos class
        pos_word_count = self.word_probs_pos.sum()
        inv_pos_word_count = 1 / pos_word_count
        self.word_probs_pos = inv_pos_word_count * self.word_probs_pos

        # same as above but for neg class
        neg_word_count = self.word_probs_neg.sum()
        inv_neg_word_count = 1 / neg_word_count
        self.word_probs_neg = inv_neg_word_count * self.word_probs_neg
    
    def predict(self, X):
        # labels to be returned
        labels = np.zeros(len(X))
        #P(Y == +1) and P(Y == -1)
        vec_len = len(X[0])

        for i in range(len(X)):
            # perform logP(Y) + SUM {i=1 to n} log(P(Xi|Y))
            prob_pos = np.log(self.prob_Y_1)
            prob_neg = np.log(1 - self.prob_Y_1)
            vector = X[i]
            for j in range(vec_len):
                # if word does not occur, skip
                if vector[j] == 0:
                    continue
                prob_pos += (vector[j]**2) * np.log(self.word_probs_pos[j])
                prob_neg += (vector[j]**2) * np.log(self.word_probs_neg[j])

            labels[i] = 1 if (prob_pos > prob_neg) else 0
        return labels

# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
        

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")
        
    
    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
