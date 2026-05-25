import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from naivni_bajes import learn, predict_batch


class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        data = X.copy()
        data['_target'] = y.values
        self.model_ = learn(data, '_target', self.smoothing)
        return self

    def predict(self, X):
        results = predict_batch(self.model_, X)
        return results['predikcija'].values


# =============================================================================
# KORISCENJE
# =============================================================================

data = pd.read_csv('../data/drug.csv')
kolona_klase = data.columns[-1]

X = data.drop(columns=[kolona_klase])
y = data[kolona_klase]

# enkoduj kategoricke kolone za sklearn
X_encoded = X.copy()
for col in X_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])

# nas model - koristi originalni X (podrzava kategoricke)
clf = NaiveBayesClassifier(smoothing=1.0)
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print(f"Nas NB      - tacnost: {scores.mean():.4f} (+/- {scores.std():.4f})")

# sklearn model - koristi enkodovani X (samo numericki)
gnb = GaussianNB()
scores_gnb = cross_val_score(gnb, X_encoded, y, cv=10, scoring='accuracy')
print(f"Sklearn GNB - tacnost: {scores_gnb.mean():.4f} (+/- {scores_gnb.std():.4f})")
