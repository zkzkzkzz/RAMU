import pandas as pd
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB


def learn(X, y, base_learners=None, ensemble_size=5, learning_rate=1.0):
    """
    AdaBoost — ucenje ansambla.

    X               : DataFrame (n×m) — atributi
    y               : Series (n,) — labele, vrednosti +1/-1
    base_learners   : lista klasa baznih algoritama (npr. [GaussianNB]).
                       Svaka mora imati .fit(X, y, sample_weight) i .predict(X).
                       Default: [GaussianNB]
    ensemble_size   : broj modela u ansamblu
    learning_rate   : skalira tezinu modela (alfa); <1 = slabija adaptacija na greske

    Vraca: (ensemble, model_weights) — lista naucenih modela i njihove tezine
    """
    if base_learners is None:
        base_learners = [GaussianNB]

    n = len(X)
    instance_weights = pd.Series(np.array([1 / n] * n), index=X.index)

    ensemble = []
    model_weights = np.zeros(ensemble_size)

    for t in range(ensemble_size):
        # slucajan izbor baznog algoritma za ovu iteraciju
        chosen_learner = base_learners[np.random.randint(len(base_learners))]
        alg = chosen_learner()
        model = alg.fit(X, y, sample_weight=instance_weights)

        predictions = model.predict(X)
        error = (predictions != y).astype(int)
        weighted_error = (error * instance_weights).sum()

        # tezina modela, skalirana learning_rate-om
        alfa = learning_rate * 0.5 * math.log((1 - weighted_error) / weighted_error)

        ensemble.append(model)
        model_weights[t] = alfa

        # update tezina instanci
        factor = np.exp(-alfa * predictions * y)
        instance_weights = instance_weights * factor
        instance_weights = instance_weights / instance_weights.sum()

    return ensemble, model_weights


def predict(X, ensemble, model_weights):
    """
    Predikcija ansambla, sa confidence-om.

    X              : DataFrame (n×m)
    ensemble       : lista naucenih modela
    model_weights  : tezine modela (alfa vrednosti)

    Vraca: DataFrame sa kolonama 'prediction' (+1/-1) i 'confidence' (0 do 1)
    """
    individual_preds = pd.DataFrame([model.predict(X) for model in ensemble]).T

    weighted_sum = individual_preds.dot(model_weights)
    prediction = np.sign(weighted_sum)

    confidence = weighted_sum.abs() / np.abs(model_weights).sum()

    return pd.DataFrame({'prediction': prediction, 'confidence': confidence}, index=X.index)


if __name__ == '__main__':
    data = pd.read_csv('data/drugY.csv')
    X = data.drop('Drug', axis=1)
    y = data['Drug'] * 2 - 1
    X = pd.get_dummies(X)

    ensemble, model_weights = learn(X, y, ensemble_size=5)

    results = predict(X, ensemble, model_weights)
    accuracy = (results['prediction'] == y).mean()
    print(f"Tacnost ansambla: {accuracy:.4f}")
    print(results.head())