import pandas as pd
import numpy as np

data = pd.read_csv('../data/drug.csv')
kolona_klase = data.columns[-1]

def is_numeric(data, col):
    return pd.api.types.is_numeric_dtype(data[col])

def learn(data, kolona_klase, smoothing):
    model = {}
    # vadimo numpy array jedinstvenih vrednosti redova iz kolone Drug
    classes = data[kolona_klase].unique()
    # punimo recnik sa jednim parom: kljuc je _classes vrednost je array vrednosti
    model['_classes'] = classes

    # log apriori verovatnoce - vrvca svake klase generalno
    apriori = (data[kolona_klase].value_counts() + smoothing) / (len(data) + smoothing * len(classes))
    apriori = apriori.replace(0, 1e-9)

    # sabiramo logaritme verovatnoca zbog underflow-a
    model['_apriori'] = np.log(apriori)

    for col in data.drop(columns=[kolona_klase]).columns:
        if is_numeric(data, col):
            # numericki atribut -> normalna raspodela, cuvamo mean i std po klasi
            stats = data.groupby(kolona_klase)[col].agg(['mean', 'std'])
            stats['std'] = stats['std'].replace(0, 1e-9)

            # ide po iterabilnoj: kljuc je ime kolone, vrednost je tuple sa tipom podataka i statom
            model[col] = ('gaussian', stats)
        else:
            # kategoricki atribut -> matrica kontigencije + smoothing
            mat = pd.crosstab(data[col], data[kolona_klase])
            # trazimo broj razlicitih vrednosti - vadi broj redova u toj koloni npr 2 za Pol
            n_values = mat.shape[0]
            mat = (mat + smoothing) / (mat.sum(axis=0) + smoothing * n_values)
            mat = mat.replace(0, 1e-9)
            model[col] = ('categorical', np.log(mat))

    return model


def predict(model, instance):
    log_probs = {}

    for cls in model['_classes']:
        log_aprior = model['_apriori'][cls]
        log_likelihood = 0

        for col, value in instance.items():
            if col not in model:
                continue

            # raspakivanje tuple-a
            attr_type, attr_data = model[col]

            # numericki
            if attr_type == 'gaussian':
                mean = attr_data.loc[cls, 'mean']
                std  = attr_data.loc[cls, 'std']
                # log gustina normalne raspodele
                log_likelihood += -0.5 * np.log(2 * np.pi * std**2) - (value - mean)**2 / (2 * std**2)

            # kategoricki
            else:
                if value in attr_data.index:
                    log_likelihood += attr_data.loc[value, cls]
                else:
                    log_likelihood += np.log(1e-9)  # za nepoznate vrednosti

        log_probs[cls] = log_aprior + log_likelihood

    prediction = max(log_probs, key=log_probs.get)
    return prediction, log_probs


def predict_batch(model, data):
    rezultati_lista = []

    # prolazimo kroz sve pacijente
    for i in range(len(data)):
        predikcija, log_posteriori = predict(model, data.iloc[i])

        # pravimo ceo red sa predikcijom i svim verovatnoćama
        novi_red = {'predikcija': predikcija}
        for cls, log_posterior_vrednost in log_posteriori.items():
            novi_red[f'log_P(klasa={cls})'] = log_posterior_vrednost

        rezultati_lista.append(novi_red)

    # pretvaramo listu u DataFrame i kombinujemo je sa originalnim podacima
    predikcije_df = pd.DataFrame(rezultati_lista, index=data.index)
    return pd.concat([data, predikcije_df], axis=1)

# Pozivanje modela
if __name__ == '__main__':
    smoothing = 1
    model = learn(data, kolona_klase, smoothing)

    novi_pacijent = {
        'Age': 25,
        'Sex': 'F',
        'BP': 'HIGH',
        'Cholesterol': 'NORMAL',
        'NaK': 15.5
    }

    predikcija, verovatnoce = predict(model, novi_pacijent)

    print(f"predikcija leka za jednog: {predikcija}")
    print(f"verovatnoce lekova za jednog: {verovatnoce}")


    results = predict_batch(model, data.drop(columns=[kolona_klase]))


    correct = (results['predikcija'] == data[kolona_klase]).sum()
    print(f"Tacnost: {correct / len(data) * 100:.2f}%")
    print(results.head(10).to_string())
