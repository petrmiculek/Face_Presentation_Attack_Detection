# stdlib
from os.path import join

# external
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix as conf_mat, classification_report, roc_curve, RocCurveDisplay, auc
import seaborn as sns


# local
# -

def compute_eer(gts_binary, probs):
    """ Calculate the Equal Error Rate (EER) and the corresponding threshold.

    :param gts_binary: ground truth binary labels
    :param probs: probabilities of the positive class
    :return: EER, threshold
    """
    fpr, tpr, thresholds = roc_curve(gts_binary, probs)
    fnr = 1 - tpr
    th = np.nanargmin(np.absolute((fnr - fpr)))
    eer_threshold = thresholds[th]
    eer = fpr[th] * 100
    return eer, eer_threshold


def compute_metrics(gts, pred_classes, bona_fide=0):
    """ Compute metrics for binary classification.

    Important: TP=True Positive = Attack classified as Attack
    It is a binary evaluation, ignoring the specific attack type!
    """
    if type(gts) == list:
        gts = np.concatenate(gts)
    if type(pred_classes) == list:
        pred_classes = np.concatenate(pred_classes)

    ''' Multi-class Metrics: Accuracy only '''
    accuracy_multiclass = np.sum(gts == pred_classes) / len(gts)

    ''' Binary Metrics: '''
    tp = np.sum(np.logical_and(pred_classes != bona_fide, gts != bona_fide))  # binary true positive
    tn = np.sum(np.logical_and(pred_classes == bona_fide, gts == bona_fide))
    fp = np.sum(np.logical_and(pred_classes != bona_fide, gts == bona_fide))
    fn = np.sum(np.logical_and(pred_classes == bona_fide, gts != bona_fide))

    ''' Accuracy '''
    accuracy_binary = (tp + tn) / (tp + tn + fp + fn)

    ''' Precision '''
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    ''' Recall '''
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    ''' F1 (micro) '''
    if precision + recall == 0:
        f1_binary = 0
    else:
        f1_binary = 2 * (precision * recall) / (precision + recall)
        # note to self:
        #   Macro F1 would do an unweighted average of per-class F1s.
        #   Weighted F1 would weigh the averages by the class support.

    ''' Attack Presentation Classification Error Rate (APCER) '''
    # == False Rejection Rate (FRR)
    if fn + tp == 0:
        apcer = 0
    else:
        apcer = fn / (tp + fn)

    ''' Bone-Fide Presentation Classification Error Rate (BPCER) '''
    # == False Acceptance Rate (FAR)
    if tn + fp == 0:
        bpcer = 0
    else:
        bpcer = fp / (fp + tn)  # false acceptance rate

    ''' Average Classification Error Rate '''
    acer = (apcer + bpcer) / 2  # average error rate

    return {
        '#TP': tp,
        '#TN': tn,
        '#FP': fp,
        '#FN': fn,
        'Accuracy': accuracy_multiclass,
        'AccuracyBinary': accuracy_binary,
        'Precision': precision,
        'Recall': recall,
        'F1Binary': f1_binary,
        'APCER': apcer,
        'BPCER': bpcer,
        'ACER': acer,
    }


def plot_confusion_matrix(gts, preds, output_path=None, labels=None, show=False, **kwargs):
    """
    Create and show/save confusion matrix

    :param gts: ground truth labels
    :param preds: predicted labels
    :param output_path: where to save the plot
    :param labels: list of labels to use
    :param show: whether to show the plot
    **keyword arguments:
    :param normalize: whether to normalize the confusion matrix
    :param title_suffix: suffix to add to the title
    """
    normalize = kwargs.pop('normalize', False)
    title_suffix = kwargs.pop('title_suffix', '')
    title = 'Confusion Matrix' + (' - ' + title_suffix if title_suffix else '')
    plot_kwargs = {}
    if normalize:
        # {'true', 'pred', 'all'}, default = None
        normalize = 'true'
        plot_kwargs.update({'vmin': 0.0, 'vmax': 1.0, 'fmt': '0.2f'})
    else:
        normalize = None
        plot_kwargs['fmt'] = 'd'

    ''' Remove classes with no samples (predicted or true) '''
    labels_idxs = np.arange(len(labels))
    if len(labels_idxs) > 2:  # don't remove for binary
        labels_idxs = labels_idxs[np.isin(labels_idxs, preds) | np.isin(labels_idxs, gts)]
        labels = np.array(labels)[labels_idxs]

    cm = conf_mat(list(gts), list(preds), normalize=normalize, labels=labels_idxs)
    ''' Print to console '''
    print(title)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(cm)
    ''' Plot '''
    sns.set_context('paper', font_scale=2.5 if len(labels) == 2 else 1.5)
    fig_cm = sns.heatmap(
        cm,
        annot=True,
        robust=True,
        xticklabels=labels,
        yticklabels=labels,
        **plot_kwargs)
    fig_cm.set_title(title)
    fig_cm.set_xlabel('Predicted')
    fig_cm.set_ylabel('True')
    fig_cm.axis('on')
    fig_cm.figure.tight_layout(pad=0.5)

    if show:
        fig_cm.figure.show()
    if output_path:
        fig_cm.figure.savefig(output_path, bbox_inches='tight')
    plt.close(fig_cm.figure)


def plot_roc_curve(gts, probs, show=True, output_path=None):
    """ Plot a Receiver Operating Characteristic (ROC) curve.

    :param gts: Ground truth binary labels.
    :param probs: probabilities of the positive class.
    :param show: display the plot
    :param output_path: Path to save the plot
    """

    fpr, tpr, thresholds = roc_curve(gts, probs)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    try:
        if show:
            plt.title('ROC Curve')
            plt.tight_layout()
            plt.show()

        if output_path is not None:
            plt.savefig(output_path, bbox_inches='tight')
    except Exception as e:
        print('Failed plotting/saving ROC curve\n', e)
