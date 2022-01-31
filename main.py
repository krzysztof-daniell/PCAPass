import argparse
from timeit import default_timer
from typing import Tuple

import numpy as np
from dgl import DGLGraph
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.metrics import accuracy_score, log_loss
from torch import Tensor
from xgboost import XGBClassifier

from pcapass import PCAPass


def process_dataset(
    dataset: str,
    dataset_root: str,
    reverse_edges: bool,
    self_loop: bool,
) -> Tuple[DGLGraph, Tensor]:
    if dataset == 'Reddit':
        dataset = RedditDataset(raw_dir=dataset_root)

        g = dataset[0]
        labels = g.ndata['label']

        train_idx = g.ndata['train_mask']
        valid_idx = g.ndata['val_mask']
        test_idx = g.ndata['test_mask']
    else:
        dataset = DglNodePropPredDataset(
            name=args.dataset, root=args.dataset_root)

        split_idx = dataset.get_idx_split()

        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        g, labels = dataset[0]

    if reverse_edges:
        src, dst = g.all_edges()

        g.add_edges(dst, src)

    if self_loop:
        g = g.remove_self_loop().add_self_loop()
    else:
        g = g.remove_self_loop()

    return g, labels, train_idx, valid_idx, test_idx


def preprocess_features(
    pcapass: PCAPass,
    g: DGLGraph,
    node_feats: Tensor,
) -> Tuple[Tensor, float]:
    start = default_timer()

    x = pcapass(g, node_feats)

    stop = default_timer()
    pcapass_time = stop - start

    return x, pcapass_time


def train(
    model: XGBClassifier,
    train_feats: Tensor,
    train_labels: Tensor,
    valid_feats: Tensor,
    valid_labels: Tensor,
    early_stopping: int = 10,
) -> float:
    start = default_timer()

    model.fit(
        train_feats,
        train_labels,
        eval_set=[(valid_feats, valid_labels)],
        early_stopping_rounds=early_stopping,
    )

    stop = default_timer()
    training_time = stop - start

    return training_time


def evaluate(
    model: XGBClassifier,
    eval_feats: Tensor,
    eval_labels: Tensor,
    xgboost_labels: Tensor,
) -> Tuple[float]:
    start = default_timer()

    logits = model.predict_proba(
        eval_feats, iteration_range=(0, model.best_iteration + 1))
    predictions = np.argmax(logits, axis=1)

    loss = log_loss(eval_labels, logits, labels=xgboost_labels)
    accuracy = accuracy_score(eval_labels, predictions)

    stop = default_timer()
    eval_time = stop - start

    return loss, accuracy, eval_time


def run(args: argparse.ArgumentParser) -> Tuple[float]:
    g, labels, train_idx, valid_idx, test_idx = process_dataset(
        args.dataset,
        args.dataset_root,
        args.reverse_edges,
        args.self_loop,
    )

    pcapass = PCAPass(
        args.khop,
        args.hidden_feats,
        seed=args.seed,
    )

    print(f'## Started PCAPass preprocessing ##')
    node_feats, pcapass_time = preprocess_features(pcapass, g, g.ndata['feat'])
    print(f'## Finished PCAPass preprocessing. Time: {pcapass_time:.2f} ##')

    train_feats = node_feats[train_idx]
    train_labels = labels[train_idx]

    valid_feats = node_feats[valid_idx]
    valid_labels = labels[valid_idx]

    test_feats = node_feats[test_idx]
    test_labels = labels[test_idx]

    xgboost_labels = np.unique(train_labels)

    model = XGBClassifier(
        n_estimators=args.num_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        verbosity=1,
        objective='multi:softprob',
        eval_metric='mlogloss',
        booster='gbtree',
        tree_method='auto',
        gamma=args.gamma,
        min_child_weight=args.min_child_weight,
        max_delta_step=args.max_delta_step,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        colsample_bylevel=args.colsample_bylevel,
        colsample_bynode=args.colsample_bynode,
        base_score=0.5,
        random_state=args.seed,
        use_label_encoder=False,
    )

    print(f'## Started XGBoost training ##')
    training_time = train(model, train_feats, train_labels,
                          valid_feats, valid_labels)
    print(f'## Finished XGBoost training. Time: {training_time:.2f} ##')

    print(f'## Started valid inference ##')
    valid_loss, valid_accuracy, valid_time = evaluate(
        model, valid_feats, valid_labels, xgboost_labels)
    print(f'## Finished valid inference. Loss: {valid_loss:.2f} '
          f'Accuracy: {valid_accuracy:.4f} Time: {valid_time:.2f} ##')

    print(f'## Started test inference ##')
    test_loss, test_accuracy, test_time = evaluate(
        model, test_feats, test_labels, xgboost_labels)
    print(f'## Finished test inference. Loss: {test_loss:.2f} '
          f'Accuracy: {test_accuracy:.4f} Time: {test_time:.2f} ##')

    return valid_accuracy, test_accuracy


def run_submission(args: argparse.ArgumentParser) -> None:
    valid_accuracies = []
    test_accuracies = []

    for seed in range(10):
        print(f'## Started seed: {seed} ##')

        args.seed = seed

        valid_accuracy, test_accuracy = run(args)

        valid_accuracies.append(valid_accuracy)
        test_accuracies.append(test_accuracy)

        print(f'## Finished seed: {args.seed} ##')

    valid_accuracy_mean = np.mean(valid_accuracies)
    valid_accuracy_std = np.std(valid_accuracies)

    test_accuracy_mean = np.mean(test_accuracies)
    test_accuracy_std = np.std(test_accuracies)

    print('## Submission results ##')
    print(f'Valid Accuracy: {valid_accuracy_mean} ± {valid_accuracy_std} '
          f'Test Accuracy: {test_accuracy_mean} ± {test_accuracy_std}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('PCAPass + XGBoost')

    argparser.add_argument('--dataset', default='ogbn-products', type=str,
                           choices=['ogbn-arxiv', 'ogbn-products', 'reddit'])
    argparser.add_argument('--dataset-root', default='dataset', type=str)
    argparser.add_argument('--reverse-edges', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--self-loop', default=True,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--khop', default=11, type=int)
    argparser.add_argument('--hidden-feats', default=100, type=int)
    argparser.add_argument('--num-estimators', default=7000, type=int)
    argparser.add_argument('--max-depth', default=6, type=int)
    argparser.add_argument('--lr', default=0.1, type=float)
    argparser.add_argument('--gamma', default=0, type=float)
    argparser.add_argument('--min-child-weight', default=1, type=float)
    argparser.add_argument('--max-delta-step', default=0, type=float)
    argparser.add_argument('--subsample', default=1, type=float)
    argparser.add_argument('--colsample-bytree', default=1, type=float)
    argparser.add_argument('--colsample-bylevel', default=1, type=float)
    argparser.add_argument('--colsample-bynode', default=1, type=float)
    argparser.add_argument('--seed', default=13, type=int)
    argparser.add_argument('--submission', default=False,
                           action=argparse.BooleanOptionalAction)

    args = argparser.parse_args()

    if args.submission:
        run_submission(args)
    else:
        run(args)
