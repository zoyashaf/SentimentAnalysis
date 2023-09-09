from keras.layers import Input, UpSampling2D,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.datasets import cifar10
import tensorflow as tf
from keras.metrics import categorical_accuracy, categorical_crossentropy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import pickle, os, zipfile, glob

    

class PseudoCallback(Callback):
    
    def __init__(self, model, n_labeled_sample, batch_size, X, Y, n_classes, padding_len = 1000):
        self.n_labeled_sample = n_labeled_sample
        self.batch_size = batch_size
        self.model = model
        self.n_classes = n_classes
        self.maxlen =padding_len
        X_train, self.X_test, y_train, self.y_test = train_test_split(X,Y,test_size=0.3, random_state=100, shuffle=True)
        
        
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        self.X_train_labeled = tf.keras.utils.pad_sequences(X_train[indices[:n_labeled_sample]], self.maxlen)
        self.y_train_labeled = y_train[indices[:n_labeled_sample]]
        self.X_train_unlabeled = tf.keras.utils.pad_sequences(X_train[indices[n_labeled_sample:]], self.maxlen)
        self.y_train_unlabeled_groundtruth = y_train[indices[n_labeled_sample:]]
        self.X_test = tf.keras.utils.pad_sequences(X_train, self.maxlen)
    
    
        self.y_train_unlabeled_prediction = np.random.randint(2, size=(self.y_train_unlabeled_groundtruth.shape[0]))
        
        self.train_steps_per_epoch = X_train.shape[0] // batch_size
        self.test_stepes_per_epoch = self.X_test.shape[0] // batch_size
        
        self.alpha_t = 0.0
    
        self.unlabeled_accuracy = []
        self.labeled_accuracy = []

    def train_mixture(self):
      
        X_train_join = np.r_[self.X_train_labeled, self.X_train_unlabeled]
        y_train_join = np.r_[self.y_train_labeled, self.y_train_unlabeled_prediction]
        flag_join = np.r_[np.repeat(0.0, self.X_train_labeled.shape[0]),
                         np.repeat(1.0, self.X_train_unlabeled.shape[0])].reshape(-1,1)
        indices = np.arange(flag_join.shape[0])
        np.random.shuffle(indices)
        return X_train_join[indices], y_train_join[indices], flag_join[indices]

    def train_generator(self):
        while True:
            X, y, flag = self.train_mixture()
            n_batch = X.shape[0] // self.batch_size
            for i in range(n_batch):
                X_batch = (X[i*self.batch_size:(i+1)*self.batch_size]).astype(np.float32)
                y_batch = to_categorical(y[i*self.batch_size:(i+1)*self.batch_size], self.n_classes)
                y_batch = np.c_[y_batch, flag[i*self.batch_size:(i+1)*self.batch_size]]
                yield X_batch, y_batch

    def test_generator(self):
        while True:
            indices = np.arange(self.y_test.shape[0])
            np.random.shuffle(indices)
            for i in range(len(indices)//self.batch_size):
                current_indices = indices[i*self.batch_size:(i+1)*self.batch_size]
                X_batch = (self.X_test[current_indices]).astype((np.float32))
                
                #(np.object)
                y_batch = to_categorical(self.y_test[current_indices], self.n_classes)
                y_batch = np.c_[y_batch, np.repeat(0.0, y_batch.shape[0])] # flagは0とする
                yield X_batch, y_batch

    def loss_function(self, y_true, y_pred):
        y_true_item = y_true[:, :self.n_classes]
        unlabeled_flag = y_true[:, self.n_classes]
        entropies = categorical_crossentropy(y_true_item, y_pred)
        coefs = 1.0-unlabeled_flag + self.alpha_t * unlabeled_flag # 1 if labeled, else alpha_t
        return coefs * entropies

    def accuracy(self, y_true, y_pred):
        y_true_item = y_true[:, :self.n_classes]
        return categorical_accuracy(y_true_item, y_pred)

    def on_epoch_end(self, epoch, logs):
       #defining alpha
        if epoch < 10:
            self.alpha_t = 0.0
        elif epoch >= 70:
            self.alpha_t = 3.0
        else:
            self.alpha_t = (epoch - 10.0) / (70.0-10.0) * 3.0
      
        self.y_train_unlabeled_prediction = np.argmax(
            self.model.predict(self.X_train_unlabeled), axis=-1,) 
        y_train_labeled_prediction = np.argmax(
            self.model.predict(self.X_train_labeled), axis=-1)
 
        self.unlabeled_accuracy.append(np.mean(
            self.y_train_unlabeled_groundtruth == self.y_train_unlabeled_prediction))
        self.labeled_accuracy.append(np.mean(
            self.y_train_labeled == y_train_labeled_prediction))
        print(f"labeled accuracy {self.labeled_accuracy[-1]}, unlabeled accuracy : {self.unlabeled_accuracy[-1]}")

    def on_train_end(self, logs):
        y_true = np.ravel(self.y_test)
        emb_model = Model(self.model.input, self.model.layers[-2].output)
        embedding = emb_model.predict(self.X_test )
        proj = TSNE(n_components=2).fit_transform(embedding)
        cmp = plt.get_cmap("tab10")
        plt.figure()
        for i in range(10):
            select_flag = y_true == i
            plt_latent = proj[select_flag, :]
            plt.scatter(plt_latent[:,0], plt_latent[:,1], color=cmp(i), marker=".")
        plt.savefig(f"result_pseudo_trans_mobile/embedding_{self.n_labeled_sample:05}.png")

