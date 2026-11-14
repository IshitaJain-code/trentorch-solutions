import numpy as np


def softmax_overflow_incident(logits: np.ndarray) -> np.ndarray:
    """
    logits: array of shape (n, d). Rows are independent.

    Returns an array of shape (n, d): the stable, max-subtracted softmax
    of each row.
    """
    # TODO: subtract each row's max before exponentiating, then
    # normalize each row by its own sum.
    row_max = np.max(logits, axis=1, keepdims=True)

    shifted = logits - row_max

    exp_values = np.exp(shifted)

    row_sums = np.sum(exp_values, axis=1, keepdims=True)

    return exp_values / row_sums
