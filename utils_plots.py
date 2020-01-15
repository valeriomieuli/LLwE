import matplotlib.pyplot as plt
import numpy as np


def train_plot(train_result, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(4)
    fig.set_figwidth(9)

    ax1.plot(train_result.epoch, train_result.history["loss"], label="train_loss", color='red')
    ax1.set(xlabel='Epochs', ylabel='Loss')

    ax2.plot(train_result.epoch, train_result.history["acc"], label="train_acc", color='blue')
    ax2.set_ylim(bottom=0, top=1)
    ax2.set(xlabel='Epochs', ylabel='Accuracy')

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(fname)


def accuracy_barchart(accuracies, phase, data_split):
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width() / 2., 1.025 * height, '%.3f' % height,
                        ha='center', va='bottom', rotation=45)

    assert phase in ['autoencoders', 'experts']
    assert data_split in ['train', 'valid', 'test']

    datasets = list(accuracies.keys())
    accuracies = [accuracies[dataset] for dataset in datasets]
    n_tasks = len(accuracies)
    x_labels = ['Task: 1:' + str(i + 2) for i in range(n_tasks - 1)]
    x = np.arange(n_tasks - 1)

    width = .1
    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(12)
    fig.tight_layout(pad=12)
    ax.set_ylim(bottom=0, top=1.1)
    ax.set_ylabel('Accuracy')
    ax.set_title(phase.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    bars = [None for _ in range(len(x) + 1)]
    if len(x) % 2 == 0:
        bars[int(n_tasks / 2)] = ax.bar(x, accuracies[int(n_tasks / 2)], width=width, label=datasets[int(n_tasks / 2)])
        for i in reversed(range(0, int(n_tasks / 2))):
            bars[i] = ax.bar(x - width * (int(n_tasks / 2 - i)), accuracies[i], width=width, label=datasets[i])
        for i in range(int(n_tasks / 2 + 1), n_tasks):
            bars[i] = ax.bar(x + width * (i - int(n_tasks / 2)), accuracies[i], width=width, label=datasets[i])
    else:
        for i in reversed(range(0, int(n_tasks / 2))):
            bars[i] = ax.bar(x - (width / 2 + width * (n_tasks / 2 - i - 1)), accuracies[i], width=width,
                             label=datasets[i])
        for i in range(int(n_tasks / 2), n_tasks):
            bars[i] = ax.bar(x + (width / 2 + width * (i - n_tasks / 2)), accuracies[i], width=width, label=datasets[i])

    for b in bars:
        autolabel(b)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.92))
    plt.savefig(data_split + '_' + phase + '_acc_barchart.jpg')
