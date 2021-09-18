import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from pr_curves import generate_test_codes_two_model, generate_test_codes_vae, get_classifier_scores
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from scipy.stats import binom
from data_preparation.data_processing_utils import min_max_norm, min_max_norm_vectorized


class MyMLPClassifier(torch.nn.Module):

    def __init__(self, in_size=256):
        super(MyMLPClassifier, self).__init__()
        self.in_size = in_size
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
                                       )

    def fit(self, train_data, train_labels, batch_size=16, max_epochs=20, lr=1e-3):
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(min_max_norm_vectorized(torch.Tensor(train_data))*2-1,
                                                 torch.Tensor(train_labels))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(max_epochs):
            for data, labels in data_loader:
                optimizer.zero_grad()
                predictions = self.mlp(data)
                loss = self.criterion(predictions.squeeze(dim=1), labels)
                loss.backward()
                optimizer.step()
            print(f"Loss: {loss.item()}")

    def predict(self, test_data, test_labels):
        dataset = torch.utils.data.TensorDataset(min_max_norm_vectorized(torch.Tensor(test_data))*2-1,
                                                 torch.Tensor(test_labels))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        predictions = []
        pred_labels = []
        with torch.no_grad():
            for data, labels in data_loader:
                predictions.append(self.mlp(data).squeeze(dim=1))
                pred_labels.append(labels)
        predictions = torch.sigmoid(torch.cat(predictions)).round()
        pred_labels = torch.cat(pred_labels)

        correct = predictions == pred_labels

        return correct.float().sum(), correct.float().mean()


def lof_scores(model1, dataloader, model2=None, save_path=None):
    if model2 is None:
        codes, labels = generate_test_codes_vae(model1, dataloader, use_res=False, plot=False)
    else:
        codes, labels = generate_test_codes_two_model(model1, model2, dataloader, use_h_labels=False,
                                                      plot=False, concat_codes=False)
    lof_scores = get_classifier_scores(codes, 'lof')
    sortidxs = np.argsort(-lof_scores)
    rank_transits(dataloader.dataset, sortidxs, save_path=save_path)

    if save_path is not None:
        pickle.dump((lof_scores, sortidxs), open(f"{save_path}/scores_n_sort_indices.pkl", "wb"))
    else:
        return lof_scores, sortidxs


def ensemble_scores(model1, dataloader, model2=None, save_path=None):
    if model2 is None:
        codes, labels = generate_test_codes_vae(model1, dataloader, use_res=False, plot=False)
    else:
        codes, labels = generate_test_codes_two_model(model1, model2, dataloader, use_h_labels=False,
                                                      plot=False, concat_codes=False)

    mahal_scores = []
    for i in range(3):
        mahal_scores.append(get_classifier_scores(codes, 'mahal'))
    mahal_scores = np.vstack(mahal_scores)
    print("Inner mahal std:", mahal_scores.std(axis=0).mean())
    mahal_scores = mahal_scores.mean(axis=0)

    lof_scores = get_classifier_scores(codes, 'lof')

    ocsvm_scores = get_classifier_scores(codes, 'ocsvm')

    if_scores = []
    for i in range(5):
        if_scores.append(get_classifier_scores(codes, 'if'))
    if_scores = np.vstack(if_scores)
    print("Inner if std:", if_scores.std(axis=0).mean())
    if_scores = if_scores.mean(axis=0)

    ensmbl_scores = (min_max_norm(mahal_scores) + min_max_norm(lof_scores) +
                       min_max_norm(ocsvm_scores) + min_max_norm(if_scores))/4.

    sortidxs = np.argsort(-ensmbl_scores)

    rank_transits(dataloader.dataset, sortidxs, save_path=save_path)

    if save_path is not None:
        pickle.dump((ensmbl_scores, sortidxs), open(f"{save_path}/scores_n_sort_indices.pkl", "wb"))
    else:
        return ensmbl_scores, sortidxs


def rank_transits(dataset, sortidxs, save_path=None):
    sorted_transits = dataset.tensors[0][sortidxs]
    sorted_ticids = dataset.tensors[1][sortidxs]
    sorted_epochs = dataset.tensors[2][sortidxs]
    sorted_periods = dataset.tensors[3][sortidxs]
    sorted_dispos = dataset.tensors[4][sortidxs]

    int2aggdisp = {0: 'PC', 1: 'KP', 2: 'CP', 3: 'V', 4: 'None', 5: 'EB', 6: 'IS', 7: 'O'}

    for t, transit in enumerate(sorted_transits):
        plt.scatter(range(transit.size()[-1]), transit)
        rank = t + 1
        disposition = int2aggdisp[sorted_dispos[t].item()]
        ticid = sorted_ticids[t].int().item()
        e4 = rank // 10000
        rank_e = rank - (e4 * 10000)
        e3 = rank_e // 1000
        rank_e -= (e3 * 1000)
        e2 = rank_e // 100
        rank_e -= (e2 * 100)
        e1 = rank_e // 10
        rank_e -= (e1 * 10)
        e0 = rank_e
        plt.title(f"Rank: {rank},   TIC {ticid},   Disposition {disposition}", fontsize=18)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        if save_path is not None:
            plt.savefig(f"{save_path}/Rank_{e4}{e3}{e2}{e1}{e0}_TIC_{ticid}_Disposition_{disposition}.png")
            plt.close()
        else:
            plt.show()


def ensemble_over_multiple_runs(run_ids=(1,2,3), ranking_path="figures/rankings/ensemble_thesis", agg=False,
                                use_agg=True):
    scores = []
    ranks = defaultdict(list)
    if use_agg:
        agg_list = [(runid, runid*11, runid*111) for runid in run_ids]
    else:
        agg_list = [''] * len(run_ids)
    for ii, i in enumerate(run_ids):
        score, sortidx = pickle.load(open(f"{ranking_path}/agg_rank_run0{i}/scores_n_sort_indices{agg_list[ii]}.pkl", "rb"))
        scores.append(score)
        for r, idx in enumerate(sortidx):
            ranks[idx].append(r+1)
    scores = np.vstack(scores)
    rank_stds = np.zeros_like(scores[0])
    rank_means = np.zeros_like(rank_stds)
    for idx, r in ranks.items():
        rank_stds[idx] = np.array(r).std()
        rank_means[idx] = np.array(r).mean()
    score_means = scores.mean(axis=0)
    score_stds = scores.std(axis=0)
    if agg:
        pickle.dump((score_means, np.argsort(-score_means)),
                    open(f"{ranking_path}/agg_rank_run0{run_ids[0]}/scores_n_sort_indices{run_ids}.pkl", "wb"))
    return score_means, score_stds, rank_means, rank_stds


def classifier_2_sample_test(dataset, scikit, test_split=0.2, use_k_fold=False, alpha=0.05):
    assert 0.1 < test_split < 0.5, "Test set size should be between 10 and 50% of available data"
    idxs = np.arange(len(dataset), dtype=int)
    k_folds = KFold(n_splits=round(1./test_split), shuffle=True)
    num_test_samples = int(np.ceil(test_split*len(dataset)))
    correct_pred = 0
    num_trials = 0
    acc = 0.
    runs_completed = 0
    for train_idx, test_idx in k_folds.split(dataset):
        np.random.shuffle(idxs)
        if use_k_fold:
            test_set = dataset[test_idx, :, :]
            train_set = dataset[train_idx, :, :]
        else:
            test_set = dataset[idxs[:num_test_samples]]
            train_set = dataset[idxs[num_test_samples:]]

        test_labels = np.concatenate((np.zeros(len(test_set), dtype=np.int32), np.ones(len(test_set), dtype=np.int32)))
        test_set = np.vstack((test_set[:, 0, :], test_set[:, 1, :]))

        train_labels = np.concatenate((np.zeros(len(train_set), dtype=np.int32), np.ones(len(train_set), dtype=np.int32)))
        train_set = np.vstack((train_set[:, 0, :], train_set[:, 1, :]))

        if scikit:
            classifier = MLPClassifier((64, 10), max_iter=500)
        else:
            classifier = MyMLPClassifier()

        classifier.fit(train_set, train_labels)
        if isinstance(classifier, MyMLPClassifier):
            correct_pred_run, acc_run = classifier.predict(test_set, test_labels)
            correct_pred += correct_pred_run
            acc += acc_run
        else:
            acc_run = classifier.score(test_set, test_labels)
            correct_pred += round(acc_run*len(test_labels))
            acc += acc_run
        num_trials += len(test_labels)
        runs_completed += 1
        if not use_k_fold:
            break
    test_chance = test_labels.mean()
    acc /= runs_completed
    correct_pred = correct_pred // runs_completed
    num_trials = num_trials // runs_completed
    p_value = 1. - binom.cdf(correct_pred, num_trials, max(1.-test_chance, test_chance))

    print(f"Classifier Accuracy of {100*acc}%, p-value: {p_value}")
    return acc, p_value


def run_c2st(data, sortidxs, threshold=0.5, test_split=0.2, alpha=0.05, data_offset=None, use_k_fold=False,
             scikit=False):
    data_sorted = data[sortidxs]
    data_size = data.size()[0]
    anomaly_idx = int(np.floor(threshold * data_size))
    if data_offset is not None:
        offset_idx = round(data_offset*data_size)
        data_sorted = data_sorted[offset_idx:]
        data_size = data_sorted.size()[0]
    regular_idxs = np.random.choice(np.arange(anomaly_idx, data_size), size=anomaly_idx, replace=False)
    assert (regular_idxs >= anomaly_idx).all(), "Regular examples must be lower in ranking than anomaly examples!"
    regular_examples = data_sorted[regular_idxs].unsqueeze(dim=1).numpy()
    anomaly_examples = data_sorted[:anomaly_idx].unsqueeze(dim=1).numpy()
    dataset = np.concatenate((regular_examples, anomaly_examples), axis=1)
    return classifier_2_sample_test(dataset, scikit, test_split=test_split, use_k_fold=use_k_fold, alpha=alpha)


if __name__ == "__main__":
    # create rankings of transits for each model checkpoint
    model1 = pickle.load(open("model_checkpoints/transit_vae_run0x.pkl", "rb"))
    model2 = pickle.load(open("model_checkpoints/residual_vae_run0x.pkl", "rb"))
    real_data = pickle.load(open("data/real_dataset.pkl", "rb"))
    real_data_loader = torch.utils.data.DataLoader(real_data, shuffle=False, batch_size=256, drop_last=False)
    lof_scores(model1, real_data_loader, model2=model2, save_path="rankings/lof/agg_rank_run0x")

    # after all individual rankings have been created combine them in an ensemble
    ensemble_over_multiple_runs(run_ids=(1,11,111,2,22,222,3,33,333,4,44,444,5,55,555), agg=True, use_agg=False,
                                ranking_path="rankings/lof")
    total_ensemble_scores, total_ensemble_idxs = pickle.load(open(
        f"rankings/lof/agg_rank_run01/scores_n_sort_indices{(1,11,111,2,22,222,3,33,333,4,44,444,5,55,555)}.pkl", "rb"))

    # perform c2st
    acc, p_value = run_c2st(real_data.tensors[0], total_ensemble_idxs, threshold=0.10, test_split=0.2,
                            alpha=0.05, data_offset=None, use_k_fold=True, scikit=True)