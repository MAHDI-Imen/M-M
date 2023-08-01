from miseval import evaluate


def compute_scores(y, y_pred):
    dice_scores = evaluate(y, y_pred, metric="Dice", multiclass=True, n_classes=4)
    IoU_scores = evaluate(y, y_pred, metric="IoU", multiclass=True, n_classes=4)
    return list(dice_scores) + list(IoU_scores)


def get_metric_scores(y, y_pred):
    n_slices = y.shape[0] // 2

    ed_labels, ed_predictions = y[:n_slices], y_pred[:n_slices]
    es_labels, es_predictions = y[n_slices:], y_pred[n_slices:]

    ED_scores = compute_scores(ed_labels, ed_predictions)
    ES_scores = compute_scores(es_labels, es_predictions)

    return ED_scores + ES_scores


def main():
    return 0


if __name__ == "__main__":
    main()
