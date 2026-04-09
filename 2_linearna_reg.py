import pandas as pd
import numpy as np

data = pd.read_csv('data/boston.csv')
target = data.columns[-1]

def normalize(X):
    mean = X.mean()
    std = X.std().replace(0, 1e-9)
    X_norm = (X - mean) / std
    return X_norm, mean, std


def add_bias(X):
    m = X.shape[0]
    return np.hstack([X, np.ones((m, 1))])


def learn(data, target_col, lam=0.0, alpha=0.01, max_iter=10000, tol=0.01, method='batch', auto_lr=False, verbose=True):
    y = data[[target_col]].to_numpy()
    X_raw = data.drop(columns=[target_col])

    X_norm, X_mean, X_std = normalize(X_raw)
    X = add_bias(X_norm.to_numpy())
    m, n = X.shape

    w = np.random.randn(1, n) * 0.01

    prev_w = None
    prev_grad = None

    for it in range(max_iter):
        pred = X @ w.T
        err = pred - y

        if method == 'sgd':
            # nasumicna instanca
            idx = np.random.randint(m)
            grad = err[idx] * X[idx].reshape(1, -1)
        else:
            # batch gradijent
            grad = (err.T @ X) / m

        # L2 regularizacija — ne penalizujemo w0 (zadnji element)
        reg = lam * w.copy()
        reg[0, -1] = 0.0
        grad = grad + reg

        # Barzilai-Borwein automatski learning rate
        if auto_lr and method == 'batch' and prev_w is not None:
            delta_w = w - prev_w
            delta_grad = grad - prev_grad
            denom = float(delta_grad @ delta_grad.T)
            if denom > 1e-12:
                alpha = abs(float(delta_w @ delta_grad.T) / denom)

        prev_w = w.copy()
        prev_grad = grad.copy()
        w = w - alpha * grad

        grad_norm = float(np.abs(grad).sum())
        loss = float(np.mean(err ** 2))

        if verbose and it % 500 == 0:
            print(f"iter={it:5d} | grad_norm={grad_norm:.6f} | MSE={loss:.4f} | alpha={alpha:.6f}")

        if grad_norm < tol:
            if verbose:
                print(f"\nKonvergencija u iteraciji {it}")
            break

    return {'w': w, 'X_mean': X_mean, 'X_std': X_std}


def predict(model, data_new):
    X_norm = (data_new - model['X_mean']) / model['X_std']
    X = add_bias(X_norm.to_numpy())
    return (X @ model['w'].T).flatten()


# Koriscenje

model = learn(data, target, lam=0.1, alpha=0.01, method='batch', auto_lr=False)

preds = predict(model, data.drop(columns=[target]))
y_true = data[target].to_numpy()
rmse = np.sqrt(np.mean((preds - y_true) ** 2))
print(f"\nRMSE: {rmse:.4f}")