
def get_weights(train_labels: list) -> dict:
    """
    Getting class weights for uber_model

    Parameters:
    -----------
    y_train - training labels

    Returns:
    --------
    weights - weights for class 0 and 1
    """
    weights = {}
    n_examples = train_labels.shape[0]
    positives = train_labels.sum()
    w_1 = n_examples/(2*positives)
    w_0 = n_examples/(2*(n_examples - positives))
    weights[1] = w_1
    weights[0] = w_0

    return weights