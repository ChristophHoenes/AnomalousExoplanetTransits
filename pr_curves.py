import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from anomaly_detection import mahalanobis_distance, local_outlier_factor, isolation_forrest, one_class_svm
from train import gather_residuals_centered
from data_preparation.data_processing_utils import min_max_norm_vectorized, min_max_norm_vectorized_np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def generate_test_codes_vae(trained_model, test_loader, use_res, plot=False):
    with torch.no_grad():
        trained_model.eval()
        trained_model = trained_model.to('cpu')
        codes = []
        labels = []
        for batch in test_loader:
            batch, res, _, _, label = batch
            data = res if use_res else batch
            reconstruction, sampled, mean, kl_loss = trained_model(data.unsqueeze(dim=1), extended_output=True)
            if plot:
                plt.scatter(range(256), batch[0], c='k')
                plt.scatter(range(256), res[0])
                plt.scatter(range(256), reconstruction[0], c='g')
                plt.show()
            codes.append((mean.detach()))
            labels.append(label)
        codes = torch.cat(codes).numpy()
        labels = torch.cat(labels).numpy()
    return codes, labels


def generate_test_codes_two_model(transit_model, res_model, test_loader,
                                  use_h_labels=False, plot=False, concat_codes=False):
    with torch.no_grad():
        transit_model.eval()
        transit_model = transit_model.to('cpu')
        res_model.eval()
        codes = []
        labels = []
        for batch in test_loader:
            batch, _, shift, scale, label = batch
            if use_h_labels:
                shift /= batch.size()[-1] / 2.
                scale = 3. / scale
                rec1, rec_fitted, sampled1, mean1, kl_loss1, fitting_loss1 = transit_model(batch.unsqueeze(dim=1),
                                                                                           scale, shift, sample=False,
                                                                                           use_h_labels=use_h_labels)
            else:
                rec1, rec_fitted, sampled1, mean1, kl_loss1, fitting_loss1 = transit_model(batch.unsqueeze(dim=1),
                                                                                           sample=False)
            residual = batch - rec_fitted.squeeze(dim=1).detach()
            residual = min_max_norm_vectorized(residual) * 2. - 1.
            if plot:
                plt.scatter(range(256), batch[0])
                plt.scatter(range(256), rec_fitted.squeeze(dim=1).detach()[0], c='g')
                plt.show()
                plt.scatter(range(256), residual[0])
                plt.show()
            reconstruction2, sampled2, mean2, kl_loss2 = res_model(residual.unsqueeze(dim=1), extended_output=True)
            codes.append(torch.hstack((mean1.detach(), mean2.detach()))) if concat_codes else codes.append(mean2.detach())
            labels.append(label)
        codes = torch.cat(codes).numpy()
        labels = torch.cat(labels).numpy()
    return codes, labels


def generate_test_codes_two_model_centered(transit_model, res_model, test_loader, plot=False):
    with torch.no_grad():
        transit_model.eval()
        transit_model = transit_model.to('cpu')
        res_model.eval()
        codes = []
        labels = []
        resloader = gather_residuals_centered(transit_model, test_loader, plot=plot, return_labels=True)
        for batch in resloader:
            data, label = batch
            _, _, mean2, _ = res_model(data.unsqueeze(dim=1), extended_output=True)
            codes.append(mean2.detach())
            labels.append(label)
        codes = torch.cat(codes).numpy()
        labels = torch.cat(labels).numpy()
    return codes, labels


def get_classifier_scores(codes, classifier, **classifier_args):
    if classifier == 'mahal':
        scores, _, _ = mahalanobis_distance(codes)
    elif classifier == 'lof':
        lof_ks = [15, 20, 25, 35, 50]
        scores = []
        for k in lof_ks:
            _, _, lof = local_outlier_factor(codes, n_neighbors=k)
            scores.append(-lof.negative_outlier_factor_)
        scores = np.stack(scores, axis=0).max(axis=0)
    elif classifier == 'ocsvm':
        _, _, ocsvm = one_class_svm(codes)
        scores = -ocsvm.score_samples(codes)
    elif classifier == 'if':
        _, _, isof = isolation_forrest(codes)
        scores = -isof.score_samples(codes)
    else:
        raise RuntimeError(f"Unknown classifier type {classifier}!")
    return scores


def average_classifier_run(codes, labels, classifier, num_runs=5):
    maps = []
    precs = []
    scores_list = []
    rec_range = np.linspace(1.,0., len(codes))
    for i in range(num_runs):
        scores = get_classifier_scores(codes, classifier)
        scores_list.append(scores)
        maps.append(average_precision_score(labels > 0, scores))
        prec, rec, thresholds = precision_recall_curve(labels > 0, scores)
        prec_exact = np.zeros(len(codes))
        r_prev = 0
        for r, rec_v in enumerate(rec[1:]):
            if rec_v == 0:
                prec_exact[r_prev:-1] = 0.0
                prec_exact[-1] = 1.0
                break
            idx = (np.abs(rec_range - rec_v)).argmin()
            prec_exact[r_prev:idx] = prec[r]
            r_prev = idx
        #t_prev = 0
        #for t, thr in enumerate(thresholds):
            #idx = (np.abs(rec_range - thr)).argmin()
            #prec_exact[t_prev:idx+1] = prec[t]
            #t_prev = t
        precs.append(prec_exact)
    maps = np.array(maps)
    precs = np.stack(precs, axis=0)
    scores_list = np.stack(scores_list, axis=0)
    std = maps.std(ddof=1) if num_runs > 1 else 0.
    return precs.mean(axis=0), maps.mean(), std, scores_list
    #return precs, maps, scores_list


def get_method_results(codes, labels, num_runs=5, raw=False):
    results = []
    rec_range = np.linspace(1., 0., len(codes))
    #for cls in ['mahal', 'lof', 'ocsvm', 'if']:
    #    if cls == 'mahal' and raw:
    #        continue
    #    results.append(average_classifier_run(codes, labels, cls, num_runs=num_runs))
    if not raw:
        results.append(average_classifier_run(codes, labels, 'mahal', num_runs=num_runs))
    p, mean, std, s = average_classifier_run(codes, labels, 'lof', num_runs=1)
    results.append((p, mean, std, np.tile(s, (num_runs, 1))))
    p, mean, std, s = average_classifier_run(codes, labels, 'ocsvm', num_runs=1)
    results.append((p, mean, std, np.tile(s, (num_runs, 1))))
    results.append(average_classifier_run(codes, labels, 'if', num_runs=num_runs))
    results = list(zip(*results))
    precisions = np.vstack(results[0])
    maps = np.array(results[1])
    map_stds = np.array(results[2])
    # calculate ensemble runs
    scores = 0
    for method in results[3]:
        scores += min_max_norm_vectorized_np(method)
    precs = []
    maps_ensemble = []
    for row in range(len(scores)):
        maps_ensemble.append(average_precision_score(labels > 0, scores[row, :]))
        prec, rec, thresholds = precision_recall_curve(labels > 0, scores[row, :])
        prec_exact = np.zeros(len(codes))
        r_prev = 0
        for r, rec_v in enumerate(rec[1:]):
            if rec_v == 0:
                prec_exact[r_prev:-1] = 0.0
                prec_exact[-1] = 1.0
                break
            idx = (np.abs(rec_range - rec_v)).argmin()
            prec_exact[r_prev:idx] = prec[r]
            r_prev = idx
        #t_prev = 0
        #for t, thr in enumerate(thresholds):
        #    idx = (np.abs(rec_range - thr)).argmin()
        #    prec_exact[t_prev:idx + 1] = prec[t]
        #    t_prev = t
        precs.append(prec_exact)
    precs = np.stack(precs, axis=0).mean(axis=0)
    maps_ensemble = np.array(maps_ensemble)
    method_precisions = np.vstack((precisions, precs))
    method_maps = np.append(maps, maps_ensemble.mean())
    method_map_stds = np.append(map_stds, maps_ensemble.std(ddof=1))
    return method_precisions, method_maps, method_map_stds, prec, rec


def average_checkpoints(model_list1, model_list2=None, centered=False, raw=False, num_runs=5, agg=True):
    if centered:
        testloader = torch.utils.data.DataLoader(
            pickle.load(open("data/alt_i.pkl", "rb")),
            batch_size=256, shuffle=False)
    else:
        testloader = torch.utils.data.DataLoader(
            pickle.load(open("data/alt_h.pkl", "rb")),
            batch_size=256, shuffle=False)
    if raw:
        codes = testloader.dataset.tensors[0].numpy()
        labels = testloader.dataset.tensors[-1].numpy()
        agg_precisions, agg_maps, agg_maps_cls_std, po, ro = get_method_results(codes, labels, num_runs=num_runs, raw=True)
        agg_maps_model_std = 0
        precision_uncert = np.zeros_like(agg_precisions)
    else:
        method_precisions = []
        method_maps = []
        method_map_stds = []
        for m, model_name in enumerate(model_list1):
            model = pickle.load(open(model_name, "rb"))
            if model_list2 is not None:
                model2 = pickle.load(open(model_list2[m], "rb"))
                if centered:
                    codes, labels = generate_test_codes_two_model_centered(model, model2, testloader, plot=False)
                else:
                    codes, labels = generate_test_codes_two_model(model, model2, testloader,
                                                                  use_h_labels=False, concat_codes=False, plot=False)
            else:
                codes, labels = generate_test_codes_vae(model, testloader, use_res=False, plot=False)
            pr, ma, mas, po, ro = get_method_results(codes, labels, num_runs=num_runs)
            method_precisions.append(pr)
            method_maps.append(ma)
            method_map_stds.append(mas)
        method_precisions = np.stack(method_precisions, axis=0)
        method_maps = np.stack(method_maps, axis=0)
        method_map_stds = np.stack(method_map_stds, axis=0)
        agg_precisions = method_precisions.mean(axis=0)
        precision_uncert = method_precisions.std(axis=0)
        agg_maps = method_maps.mean(axis=0)
        agg_maps_model_std = method_maps.std(axis=0, ddof=1)
        agg_maps_cls_std = method_map_stds.mean(axis=0)
    if raw or agg:
        return agg_precisions, precision_uncert, agg_maps, agg_maps_model_std, agg_maps_cls_std
    else:
        return method_precisions, method_maps, method_map_stds


def generate_pr_results(path, centered, num_checkpoints=3, num_runs=5, save_path=None):
    df_dict = {'precision':[], 'recall':[], 'feature':[], 'classifier':[]}
    method_names = ["Mahalanobis Distance", "Local Outlier Factor", "One-Class SVM", "Isolation Forrest", "Ensemble"]

    # Raw Data
    raw_prec, raw_uncert, raw_maps, raw_std, raw_cls_std = average_checkpoints(None, centered=centered, raw=True, num_runs=num_runs)
    for i, m in enumerate(method_names[1:]):
        prec_length = len(raw_prec[i])
        df_dict['precision'].append(raw_prec[i])
        df_dict['recall'].append(np.linspace(1., 0., prec_length))
        df_dict['classifier'].extend([m] * prec_length)
        df_dict['feature'].extend(['Raw Data'] * prec_length)

    # VAE
    ckp_vae = []
    for i in range(num_checkpoints):
        ckp_vae.append(f"{path}/vae_run0{i+1}.pkl")
    vae_prec, vae_maps, vae_map_stds = average_checkpoints(ckp_vae, centered=centered, num_runs=num_runs, agg=False)
    for j in range(num_checkpoints):
        for i, m in enumerate(method_names):
            prec_length = len(vae_prec[j][i])
            df_dict['precision'].append(vae_prec[j][i])
            df_dict['recall'].append(np.linspace(1., 0., prec_length))
            df_dict['classifier'].extend([m] * prec_length)
            df_dict['feature'].extend(['VAE'] * prec_length)

    # VAE-large
    ckp_vae2 = []
    for i in range(num_checkpoints):
        ckp_vae2.append(f"{path}/vae_double_run0{i+1}.pkl")
    vae2_prec, vae2_maps, vae2_map_stds = average_checkpoints(ckp_vae2, centered=centered, num_runs=num_runs, agg=False)
    for j in range(num_checkpoints):
        for i, m in enumerate(method_names):
            prec_length = len(vae2_prec[j][i])
            df_dict['precision'].append(vae2_prec[j][i])
            df_dict['recall'].append(np.linspace(1., 0., prec_length))
            df_dict['classifier'].extend([m] * prec_length)
            df_dict['feature'].extend(['VAE-large'] * prec_length)

    # ResidualVAE
    ckp_tra = []
    ckp_res = []
    for i in range(num_checkpoints):
        ckp_tra.append(f"{path}/transit_vae_run0{i+1}.pkl")
        ckp_res.append(f"{path}/residual_vae_run0{i+1}.pkl")
    res_prec, res_maps, res_map_stds = average_checkpoints(ckp_tra, model_list2=ckp_res, centered=centered,
                                                           num_runs=num_runs, agg=False)
    for j in range(num_checkpoints):
        for i, m in enumerate(method_names):
            prec_length = len(res_prec[j][i])
            df_dict['precision'].append(res_prec[j][i])
            df_dict['recall'].append(np.linspace(1., 0., prec_length))
            df_dict['classifier'].extend([m] * prec_length)
            df_dict['feature'].extend(['ResVAE'] * prec_length)

    df_dict['precision'] = np.hstack(df_dict['precision'])
    df_dict['recall'] = np.hstack(df_dict['recall'])
    df = pd.DataFrame.from_dict(df_dict)
    if save_path is None:
        return df
    else:
        pickle.dump(df, open(save_path, "wb"))


def generate_ap_results_table(path, centered, num_checkpoints=3, num_runs=5, save_path=None):
    df_dict = {'Local Outlier Factor': [], 'One-Class SVM': [], 'Isolation Forrest': [], 'Mahalanobis Distance': [],
               'Ensemble': []}
    method_names = ["Mahalanobis Distance", "Local Outlier Factor", "One-Class SVM", "Isolation Forrest", "Ensemble"]
    indices = ['Raw Data', 'VAE', 'VAE-large', 'ResidualVAE']

    # Raw Data
    raw_prec, raw_uncert, raw_maps, raw_std, raw_cls_std = average_checkpoints(None, centered=centered, raw=True,
                                                                               num_runs=num_runs)
    df_dict[method_names[0]].append("-")
    for i, m in enumerate(method_names[1:]):
        df_dict[m].append(f"{raw_maps[i]} +/- {raw_std}")

    # VAE
    ckp_vae = []
    for i in range(num_checkpoints):
        ckp_vae.append(f"{path}/vae_run0{i + 1}.pkl")
    vae_prec, _, vae_maps, vae_map_stds, _ = average_checkpoints(ckp_vae, centered=centered, num_runs=num_runs,
                                                                 agg=True)
    for i, m in enumerate(method_names):
        df_dict[m].append(f"{vae_maps[i]} +/- {vae_map_stds[i]}")

    # VAE-large
    ckp_vae2 = []
    for i in range(num_checkpoints):
        ckp_vae2.append(f"{path}/vae_double_run0{i + 1}.pkl")
    vae2_prec, _, vae2_maps, vae2_map_stds, _ = average_checkpoints(ckp_vae2, centered=centered, num_runs=num_runs,
                                                                    agg=True)
    for i, m in enumerate(method_names):
        df_dict[m].append(f"{vae2_maps[i]} +/- {vae2_map_stds[i]}")

    # ResidualVAE
    ckp_tra = []
    ckp_res = []
    for i in range(num_checkpoints):
        ckp_tra.append(f"{path}/transit_vae_run0{i + 1}.pkl")
        ckp_res.append(f"{path}/residual_vae_run0{i + 1}.pkl")
    res_prec, _, res_maps, res_map_stds, _ = average_checkpoints(ckp_tra, model_list2=ckp_res, centered=centered,
                                                                 num_runs=num_runs, agg=True)
    for i, m in enumerate(method_names):
        df_dict[m].append(f"{res_maps[i]} +/- {res_map_stds[i]}")

    df = pd.DataFrame.from_dict(df_dict)
    df.index = indices
    if save_path is None:
        return df
    else:
        pickle.dump(df, open(save_path, "wb"))


def pr_curves_custom_figure(df, width=25.6, height=4.8,
                            aps={"Local Outlier Factor":84.03, "One-Class SVM":78.76, "Isolation Forrest":59.76,
                                 "Mahalanobis Distance":85.12},
                            order=("Raw Data", "VAE", "VAE-large", "ResidualVAE")):
    sns.set_theme()
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.color_palette("colorblind")
    df.loc[df.feature == "ResVAE", "feature"] = "ResidualVAE"
    cls_unique = list(aps.keys())

    fig = plt.figure(figsize=(width, height))
    for c, cls in enumerate(cls_unique):
        ax = fig.add_subplot(1, len(cls_unique), c+1)
        g = sns.lineplot(data=df.loc[df["classifier"] == cls], x="recall", y="precision", hue="feature", style="feature",
                        hue_order=order, style_order=order, ci="sd")
        if c == 0:
            handles, labels = g._axes.get_legend_handles_labels()
            for handle in handles:
                handle.set_linewidth(5.0)
        g.legend().remove()
        ax.set_ylim([0., 1.])
        ax.set_xlim([0., 1.])
        ax.set_xlabel("Recall", fontsize=26)
        if c == 0:
            ax.set_ylabel("Precision", fontsize=26)
        else:
            ax.yaxis.set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_title(f"{cls}: AP={aps[cls]}%", fontsize=26)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        #frameon=False,
        ncol=4,
        fontsize=26,
        fancybox=True, shadow=True,
        bbox_to_anchor=(0.5, -0.15)
    )
    fig.tight_layout()
    fig.savefig(f"figures/pr_curves.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # generate precision recall curves
    df = generate_pr_results("model_checkpoints/alt_h", False, num_checkpoints=5, num_runs=5, save_path=None)
    pr_curves_custom_figure(df, width=6.4*4, height=4.8)

    # Generate average precision results
    df = generate_ap_results_table("model_checkpoints/alt_h", False, num_checkpoints=5, num_runs=5, save_path=None)
    print(df.to_latex())


