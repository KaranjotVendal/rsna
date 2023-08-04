import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_loss_and_acc(
    log_dir, loss_ylim=(0.0, 0.9), acc_ylim=(0.7, 1.0), save_loss=None, save_acc=None
):

    metrics = pd.read_csv(f"{log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )

    plt.ylim(loss_ylim)
    if save_loss is not None:
        plt.savefig(save_loss)

    df_metrics[["train_acc", "val_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    plt.ylim(acc_ylim)
    if save_acc is not None:
        plt.savefig(save_acc)



def show_failures(
    model,
    data_loader,
    unnormalizer=None,
    class_dict=None,
    nrows=3,
    ncols=5,
    figsize=None,
):

    failure_features = []
    failure_pred_labels = []
    failure_true_labels = []

    for batch_idx, (features, targets) in enumerate(data_loader):

        with torch.inference_mode():
            features = features
            targets = targets
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

        for i in range(features.shape[0]):
            if targets[i] != predictions[i]:
                failure_features.append(features[i])
                failure_pred_labels.append(predictions[i])
                failure_true_labels.append(targets[i])

        if len(failure_true_labels) >= nrows * ncols:
            break

    features = torch.stack(failure_features, dim=0)
    targets = torch.tensor(failure_true_labels)
    predictions = torch.tensor(failure_pred_labels)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize
    )

    if unnormalizer is not None:
        for idx in range(features.shape[0]):
            features[idx] = unnormalizer(features[idx])
    nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))

    if nhwc_img.shape[-1] == 1:
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap="binary")
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[predictions[idx].item()]}"
                    f"\nT: {class_dict[targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {predictions[idx]} | T: {targets[idx]}")
            ax.axison = False

    else:

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhwc_img[idx])
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[predictions[idx].item()]}"
                    f"\nT: {class_dict[targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {predictions[idx]} | T: {targets[idx]}")
            ax.axison = False
    return fig, axes
