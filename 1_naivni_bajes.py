import pandas as pd
import numpy as np


data = pd.read_csv('data/drug.csv')
kolona_klase = data.columns[-1]
smoothing = 1.0

def is_numeric(data, col):
    return pd.api.types.is_numeric_dtype(data[col])

def learn(data, kolona_klase, smoothing):
    model = {}
    classes = data[kolona_klase].unique()
    model['_classes'] = classes

    # log apriori verovatnoce
    apriori = data[kolona_klase].value_counts()
    apriori = apriori / apriori.sum()

    # sabiramo logaritme verovatnoca zbog underflow-a
    model['_apriori'] = np.log(apriori)

    for col in data.drop(columns=[kolona_klase]).columns:
        if is_numeric(data, col):
            # numericki atribut -> normalna raspodela, cuvamo mean i std po klasi
            stats = data.groupby(kolona_klase)[col].agg(['mean', 'std'])
            stats['std'] = stats['std'].replace(0, 1e-9)
            model[col] = ('gaussian', stats)
        else:
            # kategoricki atribut -> matrica kontigencije + smoothing
            mat = pd.crosstab(data[col], data[kolona_klase])
            # trazimo broj razlicitih vrednosti
            n_values = mat.shape[0]
            mat = (mat + smoothing) / (mat.sum(axis=0) + smoothing * n_values)
            model[col] = ('categorical', np.log(mat))

    return model


def predict(model, instance):
    log_probs = {}

    for cls in model['_classes']:
        log_p = model['_apriori'][cls]

        for col, value in instance.items():
            if col not in model:
                continue

            attr_type, attr_data = model[col]

            if attr_type == 'gaussian':
                mean = attr_data.loc[cls, 'mean']
                std  = attr_data.loc[cls, 'std']
                # log gustina normalne raspodele
                log_p += -0.5 * np.log(2 * np.pi * std**2) - (value - mean)**2 / (2 * std**2)
            else:
                if value in attr_data.index:
                    log_p += attr_data.loc[value, cls]
                else:
                    log_p += np.log(1e-9)  # za nepoznate vrednosti

        log_probs[cls] = log_p

    prediction = max(log_probs, key=log_probs.get)
    return prediction, log_probs


def predict_batch(model, data):
    results = data.copy()
    predictions = []
    log_prob_cols = {cls: [] for cls in model['_classes']}

    for i in range(len(data)):
        pred, log_probs = predict(model, data.iloc[i])
        predictions.append(pred)
        for cls in model['_classes']:
            log_prob_cols[cls].append(log_probs[cls])

    results['prediction'] = predictions
    for cls in model['_classes']:
        results[f'log_P(class={cls})'] = log_prob_cols[cls]

    return results

# Pozivanje modela

model = learn(data, kolona_klase, smoothing=1.0)

novi_pacijent = {
    'Age': 25,
    'Sex': 'F',
    'BP': 'HIGH',
    'Cholesterol': 'NORMAL',
    'NaK': 15.5
}

predikcija, verovatnoce = predict(model, novi_pacijent)

print(f"predikcija za jednog: {predikcija}")
print(f"verovatnoce lekova za jednog: {verovatnoce}")


results = predict_batch(model, data.drop(columns=[kolona_klase]))


correct = (results['prediction'] == data[kolona_klase]).sum()
print(f"Tacnost: {correct / len(data) * 100:.2f}%")
print(results.head(10).to_string())
