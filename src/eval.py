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

def compute_metrics(labels, preds, bona_fide=0):
    if type(labels) == list:
        labels = np.concatenate(labels)
    if type(preds) == list:
        preds = np.concatenate(preds)

    tp = np.sum(np.logical_and(preds != bona_fide, labels != bona_fide))
    tn = np.sum(np.logical_and(preds == bona_fide, labels == bona_fide))
    fp = np.sum(np.logical_and(preds != bona_fide, labels == bona_fide))
    fn = np.sum(np.logical_and(preds == bona_fide, labels != bona_fide))

    ''' Accuracy '''
    accuracy = (tp + tn) / (tp + tn + fp + fn)

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

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    ''' Attack Presentation Classification Error Rate (APCER) '''
    # == False Rejection Rate (FRR)
    if fn + tp == 0:
        apcer = 0
    else:
        apcer = fn / (fn + tp)

    ''' Bone-Fide Presentation Classification Error Rate (BPCER) '''
    # == False Acceptance Rate (FAR)
    if tn + fp == 0:
        bpcer = 0  # specificity
    else:
        bpcer = fp / (fp + tn)  # false acceptance rate

    ''' Average Classification Error Rate '''
    acer = (apcer + bpcer) / 2  # average error rate

    return {
        '#TP': tp,
        '#TN': tn,
        '#FP': fp,
        '#FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'APCER': apcer,
        'BPCER': bpcer,
        'ACER': acer,
    }



def confusion_matrix(gts, predictions_hard, output_location=None, labels=None, show=True, **kwargs):
    """Create and show/save confusion matrix"""
    # if labels is None:
    #     # labels = list(map(str, range(np.max(gts) + 1)))
    #     labels = list(range(np.max(gts) + 1))
    # Problem with labels: data might not include all values

    model_name = kwargs.pop('model_name', 'UnnamedModel')
    normalize = kwargs.pop('normalize', False)
    title_suffix = kwargs.pop('title_suffix', '')

    title = 'Confusion Matrix' + (' - ' + title_suffix if title_suffix else '')
    epochs_trained = '20'
    plot_kwargs = {}
    if normalize:
        # {'true', 'pred', 'all'}, default = None
        normalize = 'true'
        plot_kwargs.update({
            'vmin': 0.0,
            'vmax': 1.0,
            'fmt' : '0.2f',
        })
    else:
        normalize = None
        plot_kwargs['fmt'] = 'd'

    labels_numeric = np.arange(len(labels))
    cm = conf_mat(list(gts), list(predictions_hard), normalize=normalize, labels=labels_numeric)

    # also print confusion matrix
    print(title)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm)

    sns.set_context('paper', font_scale=1.0)
    fig_cm = sns.heatmap(
        cm,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        # fmt='0.2f',
        # vmin=0.0,
        # vmax=1.0
        **plot_kwargs
    )
    fig_cm.set_title(title)
    fig_cm.set_xlabel('Predicted')
    fig_cm.set_ylabel('True')
    fig_cm.axis('on')
    fig_cm.figure.tight_layout(pad=0.5)

    if show:
        fig_cm.figure.show()

    if output_location:
        fig_cm.figure.savefig(join(output_location, 'confusion_matrix' + '.pdf'),
                              bbox_inches='tight')

    # plt.close(fig_cm.figure)

    return fig_cm


'''
# currently unused
def plot_roc_curve(gts, predictions, show=True, output_location=None):
    sns.set_context('paper', font_scale=1.8)

    fpr, tpr, thresholds = roc_curve(gts, predictions)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    try:
        if show:
            plt.title('ROC Curve')
            plt.tight_layout()
            plt.show()

        if output_location is not None:
            plt.savefig(output_location, bbox_inches='tight')
    except Exception as e:
        print('Failed plotting/saving ROC curve\n', e)
'''
