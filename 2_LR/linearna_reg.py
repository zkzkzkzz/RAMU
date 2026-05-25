import pandas as pd
import numpy as np

data = pd.read_csv('../data/boston.csv')
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

    # m - broj primera - 506
    # n - broj obelezja - 13

    w = np.random.randn(1, n) * 0.01

    prev_w = None
    prev_grad = None
    alpha_0_sgd = alpha

    for it in range(max_iter):
        pred = X @ w.T
        err = pred - y

        if method == 'sgd':
            # nasumicna instanca - SGD
            idx = np.random.randint(m)
            grad = err[idx] * X[idx].reshape(1, -1)
            alpha = alpha_0_sgd / (1 + 0.001 * it)
        else:
            # batch gradijent
            grad = (err.T @ X) / m

        # L2 regularizacija — ne penalizujemo w0 (zadnji element)
        reg = 2 * lam * w.copy()
        reg[0, -1] = 0.0
        grad = grad + reg

        # Barzilai-Borwein automatski learning rate
        if auto_lr and method == 'batch' and prev_w is not None:
            delta_w = w - prev_w
            delta_grad = grad - prev_grad
            denom = float((delta_grad @ delta_grad.T).item())

            # denominator tj imenilac
            if denom > 1e-12:
                alpha_bb = abs(float((delta_w @ delta_grad.T).item()) / denom)
                alpha = min(alpha_bb, 1.0)

        prev_w = w.copy()
        prev_grad = grad.copy()

        # tek sad azuriramo tezine
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
    # normalizujemo nove podatke i koristimo iste mean i std iz treninga
    X_norm = (data_new - model['X_mean']) / model['X_std']

    X = add_bias(X_norm.to_numpy())
    return (X @ model['w'].T).flatten()


# Koriscenje

# model = learn(data, target, lam=0, alpha=0.01, method='batch', auto_lr=True, verbose=True)
#
# preds = predict(model, data.drop(columns=[target]))
# y_true = data[target].to_numpy()
# rmse = np.sqrt(np.mean((preds - y_true) ** 2))
# print(f"\nRMSE: {rmse:.4f}")
# print(preds[4])
# print(y_true[4])

novi_primer = pd.DataFrame([{
    'CRIM': 0.00632, 'ZN': 18.0, 'INDUS': 2.31, 'CHAS': 0,
    'NOX': 0.538, 'RM': 6.575, 'AGE': 65.2, 'DIS': 4.09,
    'RAD': 1, 'TAX': 296.0, 'PTRATIO': 15.3, 'B': 396.9, 'LSTAT': 4.98
}])

model_batch = learn(data, target, lam=0.1, alpha=0.01, method='batch', auto_lr=True, verbose=False)
model_sgd   = learn(data, target, lam=0.1, alpha=0.01, method='sgd', verbose=False)

print(f"Batch predikcija: {predict(model_batch, novi_primer)[0]:.2f}")
print(f"SGD predikcija:   {predict(model_sgd, novi_primer)[0]:.2f}")
print(f"Stvarna vrednost: 24.0")
