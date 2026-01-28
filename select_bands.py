import numpy as np


def OPBS(x: np.ndarray, num_bs: int):
    """
    Ref:
    W. Zhang, X. Li, Y. Dou, and L. Zhao, "A geometry-based band
    selection approach for hyperspectral image analysis," IEEE Transactions
    on Geoscience and Remote Sensing, vol. 56, no. 8, pp. 4318–4333, 2018.
    """
    rows, cols, bands = x.shape
    eps = 1e-9

    x_2d = np.reshape(x, (rows * cols, bands))
    y_2d = x_2d.copy()
    h = np.zeros(bands)
    band_idx = []

    idx = np.argmax(np.var(x_2d, axis=0))
    band_idx.append(idx)
    h[idx] = np.sum(x_2d[:, band_idx[-1]] ** 2)

    i = 1
    while i < num_bs:
        id_i_1 = band_idx[i - 1]

        _elem, _idx = -np.inf, 0
        for t in range(bands):
            if t not in band_idx:
                y_2d[:, t] = y_2d[:, t] - y_2d[:, id_i_1] * (np.dot(y_2d[:, id_i_1], y_2d[:, t]) / (h[id_i_1] + eps))
                h[t] = np.dot(y_2d[:, t], y_2d[:, t])

                if h[t] > _elem:
                    _elem = h[t]
                    _idx = t

        band_idx.append(_idx)
        i += 1

    band_idx = sorted(band_idx)
    return band_idx


def SIFDR(pre: np.ndarray, bandnum: int) -> np.ndarray:
    """
    Real-time unsupervised hyperspectral band selection via spatial-spectral information fusion based downscaled region.

    Parameters:
    pre (np.ndarray): Raw image data after bicubic interpolation, shape (Np, Nb)
                      where Np is the number of pixels and Nb is the number of bands.
    bandnum (int): Number of band requests.

    Returns:
    np.ndarray: Band index of selected bands.
    """
    Np, Nb = pre.shape

    R = (1 / Np) * (pre.T @ pre)
    d = pre.T
    X = np.linalg.inv(R)
    group = int(np.ceil(Np / 1000))
    w = []

    for i in range(group):
        up = min((1000 * (i + 1)), Np)
        do = 1000 * i
        di = d[:, do:up]
        regularization = 1e-10
        wi = (X @ di) @ np.linalg.inv(di.T @ X @ di + regularization * np.eye(di.shape[1]))
        w.append(wi)

    w = np.abs(np.concatenate(w, axis=1))
    rho = np.mean(w, axis=1)

    Dist_matrix = G_D(pre) / np.sqrt(Np)
    Dist_matrix = np.abs(np.tanh(Dist_matrix ** 1))
    ordrho = np.argsort(rho)[::-1]
    delta = np.zeros(Nb)
    delta[ordrho[0]] = -1
    maxD = np.max(Dist_matrix)

    for i in range(1, Nb):
        delta[ordrho[i]] = maxD
        for j in range(i):
            if Dist_matrix[ordrho[i], ordrho[j]] < delta[ordrho[i]]:
                delta[ordrho[i]] = Dist_matrix[ordrho[i], ordrho[j]]

    delta[ordrho[0]] = np.max(delta)
    rho = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))

    phi = rho * delta

    order_band = np.argsort(phi)[::-1]
    C = order_band[:bandnum]
    return C


def G_D(X: np.ndarray) -> np.ndarray:
    """
    Compute the distance matrix for normalization.

    Parameters:
    X (np.ndarray): Input data.

    Returns:
    np.ndarray: Normalized distance matrix.
    """
    return np.sqrt(L2_distance(X))


def L2_distance(a: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distance.

    Parameters:
    a (np.ndarray): Input data.

    Returns:
    np.ndarray: Squared distance matrix.
    """
    sm = np.ones((1, a.shape[0]))
    aa = sm @ (a * a)
    ab = (a.T @ a)
    aa = np.round(10000 * aa) / 10000
    ab = np.round(10000 * ab) / 10000
    d = np.tile(aa.T, (1, aa.shape[1])) + np.tile(aa, (aa.shape[1], 1)) - 2 * ab
    d = np.real(d)
    d = np.maximum(d, 0)
    return d
