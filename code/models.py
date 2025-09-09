from multiprocessing import Pool

from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor as knn

import numpy as np

class Kmers():
    def __init__(self, k):
        self.k = k
        self.kmers = {}


    def fit(self, sequences, labels):
        #kmerize all sequences
        with Pool(8) as p:
            all_kmers = p.map(self.get_kmers, sequences)

        #fill dict with info from each sequence
        for kmers, l in zip(all_kmers, labels):
            for kmer in kmers:
                entry = self.kmers.get(kmer)
                if entry is None:
                    self.kmers[kmer] = (l, 1) # (label, count)
                else:
                    self.kmers[kmer] = (entry[0] + l, entry[1] + 1)
        #average phs
        for kmer in self.kmers.keys():
            (ph, counter) = self.kmers[kmer]
            self.kmers[kmer] = round(ph / counter, 2)


    def predict(self, sequences):
        #kmerize all sequences
        with Pool(8) as p:
            all_kmers = p.map(self.get_kmers, sequences)
        
        predictions = []
        for kmers in all_kmers:
            ph = 0
            count = 0
            for kmer in kmers:
                if self.kmers.get(kmer) is not None:
                    ph += self.kmers.get(kmer)
                    count += 1

            if count:
                ph /= count
                ph = round(ph, 1)
                predictions.append(ph)
            else:
                predictions.append(-1)

        return np.array(predictions)


    def test(self, sequence):
        pass

    def get_train_kmers(self):
        return self.kmers

    def calc_coverage(self, sequence):
        seq_kmers = self.get_kmers(sequence)
        all_kmers = set(self.kmers.keys())
        return len(set(seq_kmers).intersection(all_kmers)) / len(seq_kmers)

    def get_kmers(self, sequence):

        amino_acids = 'LCAGSTPFWEDNQKHRVIMY'
        kmers = []
        for i in range(len(sequence)+1-self.k):
            kmer = sequence[i:i+self.k]
            kmers.append(kmer)

        return kmers

class Dummy:
    def __init__(self, *args, **kwargs):
        pass
    def fit(self, X_train, y_train):
        self.median_pH = np.median(y_train.flatten())
    def predict(self, X_train):
        print(self.median_pH)
        return np.array([self.median_pH] * len(X_train))

models = {
    "xgboost": XGBRegressor,
    "knn": knn,
    "kmers":Kmers,
    "dummy":Dummy
}


def get_model(*args, **kwargs):

    print(kwargs)
    model_type = kwargs.get("type")
    model_params = kwargs.get("params")

    if model_type is None:
        raise ValueError("No model type passed")
    if model_params is None:
        raise ValueError("No model params passed")
    print(model_params)
    return models[model_type](**model_params)
