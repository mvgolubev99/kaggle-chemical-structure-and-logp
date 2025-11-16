import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def plot_target_distributions(target, **ids):
    """
    Draws violin plots comparing target distributions for each fold's train/test
    subsets and the final test set.

    usage example:

    plot_target_distributions(
        target = data['logp'],
        train=train_ids,
        val=val_ids,
        test=test_ids,
    )

    """
    #plt.figure(figsize=(4*len(ids),len(ids)))
    plt.figure()
    data = []
    labels = []
    for name, data_ids in ids.items():
        data.append(target.iloc[data_ids].values)
        labels.append(name)

    plt.violinplot(data, showmedians=True)
    plt.xticks(range(1, len(ids) + 1), labels=ids.keys())
    #plt.xticklabels(ids.keys(), rotation=20)
    plt.tight_layout()
    plt.title(f"target distribtuion in {', '.join(ids.keys())} subsets")
    plt.show()

def make_k_fold_split_ids(
    data_ids,
    n_splits=5,
    random_state=42,
    shuffle=True,
    ):
    """
    This function returns list of folds, where
    each fold contains indices of 'train' and 'test' subset.
    """
    splitter = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    folds_ids = []
    # No list comprehension for better readability.
    # Do not be confused! Unlike train_test_split,
    # KFold.split() method return indices of 'train' and 'test',
    # not the actual data
    for train_ids, test_ids in splitter.split(data_ids):
        fold_ids = {
            'train': data_ids[train_ids],
            'test': data_ids[test_ids],
        }
        folds_ids.append(fold_ids)
    return folds_ids

def plot_target_distributions_cv(target, folds_ids):
    """
    Draws violin plots comparing target distributions for each fold's train/test
    subsets and the final test set.
    """
    n_splits = len(folds_ids)
    fig, axes = plt.subplots(1, n_splits, figsize=(4 * n_splits, 4), sharey=True)

    if n_splits == 1:
        axes = [axes]

    for i, fold in enumerate(folds_ids):
        data = [
            target.iloc[fold['train']],
            target.iloc[fold['test']],
        ]
        labels = ['fold_train', 'fold_cv']
        axes[i].violinplot(data, showmedians=True)
        axes[i].set_xticks(range(1, len(labels) + 1))
        axes[i].set_xticklabels(labels, rotation=20)
        axes[i].set_title(f'Fold {i+1}')

    fig.suptitle('Target distributions per fold', y=1.02)
    plt.tight_layout()
    plt.show()