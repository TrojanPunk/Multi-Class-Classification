import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from scipy.sparse import csr_matrix, hstack, vstack

class classifier_chain(object):
    """
    __init__ : Constructor class
    Classifier_add : Adding a classifier layer
    Classifier_replace : Replacing a classifier layer
    Multi_to_Binary : A layer which converts Multi-label to Binary-label
    Cal_weight : A layer which return the weight
    Cal_weight_chain : A layer which return the weight chaining
    Batch_gen : Return a python generator of training batches
    Train_single : Training a single layer
    Pred_single : Predicting a single layer
    Gen_data : Generating data for the next classifier in the chain
    Fit : Training the entire chain
    """

    def __init__(self, classifier, num_labels, name, optimizers=None, losses=None, create_missing=False):
        """
        Constructor method that is used to create the chain of classifiers that will be used

        Parameters:
        classifier - The first classifier that is present
        num_labels - Number of labels in the multi-label classifier
        name - name of the classifier chain
        optimizers
        losses
        create_missing - If set to True classifiers are automatically created with configurations inferred from the input classifier.
        """

        self.classifier = classifier
        self.num_labels = num_labels
        self.classifier = []
        self.classifiers.append(self.classifier)
        self.classifier_config = self.classifier.get_config()
        self.classifier_input_shape = self.classifier_config['layers'][0]['config']['batch_input_shape']

        self.name = name
        if create_missing:
            self.optimizers = optimizers
            self.losses = losses
            for i in range(1, self.num_labels):
                cfg = self.classifier_config.copy()
                cfg['layers'][0]['config']['batch_input_shape'] = \
                    (self.classifier_input_shape[0], self.classifier_input_shape[1] + i)
                self.classifiers.append(keras.models.Sequential.from_config(cfg))
                self.classifiers[i].compile(optimizer=self.optimizers[i - 1], loss=self.losses[i - 1])
    

    def Classifier_add(self, filename):
        """ 
        This class basically loads and adds or appends a previously trained layer and adds it to self.classifier

        Parameters:
        filename - Filename that holds the classifier
        """

        if len(self.classifier) == self.num_labels:
            print("Number of classifiers are equal to the number of labels.")
        else:
            self.classifiers.append(keras.models.load_model(filename))

    def Classifier_replace(self, i, classifier):
        """
        Replace the i'th classifier in our chain

        Parameters:
        i - The index of the classifier in the chain
        classifier - The replacement classifier 
        """
        self.classifiers[i] = classifier

    @staticmethod
    def Multi_to_Binary(y, i):
        """
        Converting multi to binary

        Parameters:
        y - Multi-label
        i - Index that has to be considered
        """
        ones = np.ones(y.shape[0])
        zeros = np.zeros(y.shape[0])
        M1 = np.column_stack((ones.T, zeros))
        M2 = np.column_stack((zeros.T, ones))
        y_i = y[:, i]
        d_i = ones - y_i
        yret = (M1.T * y_i).T + (M2.T * d_i).T
        return yret

    @staticmethod
    def Cal_weight(y, debalancing=0):
        """
        Returns weight in sample weight

        Parameters:
        y - Multi-label
        debalancing - To add more features to frequently occuring labels
        """
        unique_rows, inverse, counts = np.unique(y, axis=0, return_inverse=True, return_counts=True)
        weights = np.array([float(y.shape[0]) / float(counts[m]) + debalancing for m in inverse])
        weights = weights / np.amax(weights)
        return weights

    @staticmethod
    def Cal_weight_chain(X, y, preds_start_index, debalancing=0):
        """
        This method allows the classifier chain to automatically infer class-weights during training (in chain mode).

        Parameters:
        X - Sparse matrix
        y - Sample label
        pred_start_index - The index of the first column in X where predictions have been made
        debalancing - Adds importance to more frequent labels (during training).
        """
        y = csr_matrix(y)
        data = hstack([X, y]).tocsr()
        preds_and_label = data[:, preds_start_index:].toarray()
        weights = classifier_chain.Cal_weight(preds_and_label, debalancing=debalancing)
        return weights
    
    @staticmethod
    def Batch_gen(X, y_input, batch_size=32, shuffle=False, weights_mode=None, predefined_weights=None, preds_start_index=None, debalancing=0):
        """
        Creates a python generator of training batches.

        Parameters: 
        X - Sparse matrix
        y_intput - ndarray
        batch_size - Number of samples in each size
        shuffles - whether the samples should be shuffled before creating the generator
        weights_mode - If this is the "chain" input the the weights are computed by Cal_weight_chain
        predefined_weights - Weights for the samples
        preds_start_index - Refers to the index of the first column corresponding to previous predictions.
        debalancing - Adds importance to frequently occurring labels.

        """
        X_copy = X.copy()
        y_copy = csr_matrix(np.copy(y_input))
        if weights_mode == 'chain':
            if preds_start_index is None:
                preds_start_index = X_copy.shape[1]

            weights = classifier_chain.Cal_weight_chain(
                X_copy, y_copy, preds_start_index=preds_start_index, debalancing=debalancing)
            weights = csr_matrix(weights)

        elif predefined_weights is not None:
            weights = csr_matrix(predefined_weights)
        else:
            weights = csr_matrix(np.ones(X.shape[0]))
        if shuffle:
            data = hstack([X_copy, y_copy, weights.transpose()])
            data = data.tocsr()
            row_indices = np.arange(X_copy.shape[0])
            np.random.shuffle(row_indices)
            data = data[row_indices]
            X_copy = data[:, :-3]
            y_copy = data[:, -3:-1]
            weights = data[:, -1].toarray().flatten()

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_copy[i: i + batch_size, :].toarray()
            y_batch = y_copy[i: i + batch_size, :].toarray()
            weights_batch = weights[i: i + batch_size]

            if weights_mode == 'chain':

                yield (X_batch, y_batch, weights_batch)

            elif predefined_weights is not None:
                yield (X_batch, y_batch, weights_batch)

            else:
                yield (X_batch, y_batch)
    @staticmethod
    def Train_single(self, i, X, y, epochs=1, batch_size=32, verbose=1, shuffle=True, weights_mode=None, predefined_weights=None, preds_start_index=None, debalancing=0, save_after=None):
        """
        Train a single classifier in the chain.

        Parameters:
        i : Number of the classifier in the chain (we count from 0)
        X : Samples (with possibly previous predictions appended as columns)
        y : class labels
        epochs : optional, default=1. Number of training epochs.
        batch_size : optional, default = 32. Number of samples per gradient update.
        verbose : optional, default = 1. This argument is passed to classifier.fit_generator
        shuffle : optional, default True. Whether to shuffle the training set at the start of each epoch.
        weights_mode : optional, default None. If set to 'chain' samples will automatically be weighted by the method self.Cal_weight_chain.
        predefined_weights : optional, default = None. Weights for the samples. These weights willonly be considered if weights_mode is None.
        preds_start_index : optional, default = None. Only relevant if weights_mode is chain in which case it indicates the index of the first column of X corresponding to previous predictions.
        debalancing : optional, default 0. Only relevant if weights_mode is chain in which casethe parameter adds importance to frequently occurring labels.
        save_after : optional, default None. If set to 'epoch classifier' i will be saved in its entiretyafter every epoch. If set to 'completion' we save the classifier after the method is complete.
        """

        steps_per_epoch = int(np.ceil(X.shape[0] / batch_size))
        y_input = self.project_to_binary(y, i)
        # todo: rewrite the batch generator so it restarts when all batches have been yielded, in that way we don't
        #  need to loop over the epochs, and can use the epochs parameter in keras.Sequential.fit_generator instead.
        for epoch in range(epochs):
            print('Training classifier %d: Epoch %d/%d' % (i, epoch + 1, epochs))
            batch_generator = self.Batch_gen(
                X, y_input, batch_size=batch_size, shuffle=shuffle, weights_mode=weights_mode,
                predefined_weights=predefined_weights, preds_start_index=preds_start_index,
                debalancing=debalancing)

            self.classifiers[i].fit_generator(
                generator=batch_generator, steps_per_epoch=steps_per_epoch, verbose=verbose)
            if save_after == 'epoch':
                self.classifiers[i].save('%s_classifier_%s_epoch_%s.h5' % (self.name, str(i), str(epoch)))
        
        if save_after == 'completion':
            if epochs > 0:
                self.classifiers[i].save('%s_classifier_%s_epochs_%s.h5' % (self.name, str(i), str(epochs)))

### Might have to add static 
    def Pred_single(X, classifier, batch_size=32):
        """
        The input classifier makes predictions based on the input csr-matrix X.

        """
        y = np.zeros(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i: i + batch_size, :].toarray()
            y[i: i + batch_size] = (np.argmax(classifier.predict(X_batch, batch_size=batch_size), axis=1) == 0).astype(int)

        return y

### Might have to add static 
    def Gen_data(X, classifier, batch_size=32):
        """
         Creates the next sample in the chain classifier.
        """
        preds = csr_matrix(classifier_chain.Pred_single(X=X, classifier=classifier, batch_size=batch_size))
        X_next = hstack([X, preds.transpose()]).tocsr()
        return X_next

### Might have to add static 
    def Fit(self, X, y, epochs=1, batch_size=32, verbose=1, weights_mode=None, predefined_weights=None, debalancing=0, shuffle=True, save_after='classifier'):
        """
        Trains all classifiers in the chain
        """
        if len(self.classifiers) < self.num_labels:
            raise IndexError('Method fit can only be run when there are as many classifiers as labels')
        if type(epochs) == int:
            epochs_list = self.num_labels * [epochs]
        else:
            epochs_list = epochs

        if type(debalancing) == int:
            debalancing_list = self.num_labels * [debalancing]
        else:
            debalancing_list = debalancing

        F = X.copy()
        preds_start_index = X.shape[1]
        for i in range(self.num_labels):
            if save_after == 'classifier':
                save_after = 'completion'
            elif save_after == 'epoch':
                save_after = 'epoch'

            current_mode = weights_mode
            current_weights = None

            if predefined_weights is not None:
                if i < len(predefined_weights):
                    current_weights = predefined_weights[i]
                    current_mode = None
                else:
                    predefined_weights = None

            self.fit_single(
                i=i, X=F, y=y, epochs=epochs_list[i], batch_size=batch_size, shuffle=shuffle,
                verbose=verbose, weights_mode=current_mode, predefined_weights=current_weights,
                preds_start_index=preds_start_index, debalancing=debalancing_list[i], save_after=save_after)

            if i == self.num_labels - 1:
                break