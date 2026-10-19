import numpy as np

def label_sharing_D(task, split, frac=0.3):
    """Training-free data-side predictor. D(k) = sum_c | sum_{training equations (a,b)->c} exp(2 pi i k g(a) / n) |^2,
    with g the task's group map (identity for add/sub, discrete log for mul). Returns log D for k = 1..nf.
    Equivalently: the Fourier power of N[a,a'] = number of answers shared by a and a' in the training set."""
    tr, _ = task.split(split, frac)
    a = task.a[tr].numpy(); c = task.y[tr].numpy(); ga = task.log[a]
    w = 2 * np.pi * np.arange(1, task.nf + 1) / task.n
    S = np.zeros((task.nf, task.p), complex)
    np.add.at(S.T, c, np.exp(1j * np.outer(w, ga)).T)
    return np.log((np.abs(S) ** 2).sum(1))

def attention_path_score(path_share_at_init):
    """Init-side predictor: log Fourier share of W_E W_V W_O at initialization (from train_batch record step 0)."""
    return np.log(path_share_at_init)
