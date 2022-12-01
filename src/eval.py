from os.path import join

import numpy as np
import torch
from matplotlib import pyplot as plt


from sklearn.metrics import confusion_matrix as conf_mat, classification_report, roc_curve, RocCurveDisplay, auc
import seaborn as sns


def compute_metrics(gts, predictions):
    # todo choose where to concat list into tensor
    gts = torch.cat(gts)
    predictions = torch.cat(predictions)

    # todo unified call for evaluation
    predictions_hard = torch.tensor([1 if x >= 0.5 else 0 for x in predictions])

    # print(classification_report(gts, predictions_hard))

    # unused
    # confusion_matrix(gts, predictions_hard)

    results = {}
    for metric in [accuracy]:
        results.update(metric(gts, predictions, predictions_hard))
    return results


def accuracy(gts, predictions, predictions_hard=None):
    """
    Compute metrics from predictions and ground truth
    :param gts: ground truth
    :param predictions: predictions
    :param predictions_hard: prediction decisions
    """
    if type(gts) == list:
        gts = torch.cat(gts).cpu()
        predictions = torch.cat(predictions).cpu()

    if predictions_hard is None:
        # predictions_hard = torch.tensor([1 if x >= 0.5 else 0 for x in predictions])
        predictions_hard = (predictions >= 0.5)

    correct = predictions_hard == gts

    accuracy = torch.sum(correct) / len(correct)

    results = {
        'accuracy': accuracy.item()
    }
    return results



def confusion_matrix(gts, predictions_hard, output_location=None, labels=None, show=True, normalize=True):
    """Create and show/save confusion matrix"""
    if labels is None:
        labels = list(map(str, range(np.max(gts) + 1)))

    model_name = 'LSTM_Base'  # make it a parameter
    epochs_trained = '20'
    kwargs = {}
    if normalize:
        # {'true', 'pred', 'all'}, default = None
        normalize = 'true'
        kwargs['vmin'] = 0.0
        kwargs['vmax'] = 1.0
        kwargs['fmt'] = '0.2f'
    else:
        normalize = None
        kwargs['fmt'] = 'd'

    cm = conf_mat(list(gts), list(predictions_hard), normalize=normalize)

    sns.set_context('paper', font_scale=1.8)
    fig_cm = sns.heatmap(
        cm,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        # fmt='0.2f',
        # vmin=0.0,
        # vmax=1.0
        **kwargs
    )
    fig_cm.set_title('Confusion Matrix')
    fig_cm.set_xlabel('Predicted')
    fig_cm.set_ylabel('True')
    fig_cm.axis('on')
    fig_cm.figure.tight_layout(pad=0.5)

    if show:
        fig_cm.figure.show()

    if output_location:
        fig_cm.figure.savefig(join(output_location, 'confusion_matrix' + '.pdf'),
                              bbox_inches='tight')

    plt.close(fig_cm.figure)


'''

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
