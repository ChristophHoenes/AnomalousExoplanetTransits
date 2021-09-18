import json
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from architectures import CustomVAE, TransitShapeVAE
from data_preparation.data_processing_utils import min_max_norm_vectorized, horizontal_scale_n_shift


def loss_shape_fitting(model, batch, labels, writer, iterations, gamma, beta=1., verbose=False, **kwargs):

    reconstruction, sampled, mean, kl_loss = model(batch, extended_output=True)

    shape_loss = nn.functional.mse_loss(reconstruction, labels[0].unsqueeze(dim=1))
    writer.add_scalar("Rec_Loss/train", shape_loss.item(), iterations)

    # monotonicly decreasing reconstruction weight
    mse_estimate = gamma**2
    gamma = np.sqrt(min(mse_estimate, shape_loss.detach().item()))
    rec_loss = shape_loss / (2 * gamma ** 2)

    loss = rec_loss + beta * kl_loss

    writer.add_scalar("KL_Loss/train", kl_loss.item(), iterations)
    writer.add_scalar("Gamma/train", gamma, iterations)
    writer.add_scalar("Loss/train", loss.item(), iterations)

    if verbose:
        print(f"Iteration {iterations}, Loss: {loss.item()}, Rec: {shape_loss.item()}, Gamma: {gamma}, "
              f"KL: {kl_loss.item()}")

    return loss, gamma, {'rec':reconstruction, 'z_sampled':sampled, 'z_mean':mean, 'transit_model':labels[0]}


def plot_shape_fitting(data, **kwargs):
    reconstruction = kwargs['rec']
    transit_model = kwargs['transit_model']

    size = range(len(data[0][0]))
    transit_label = transit_model[0].cpu()
    plt.plot(size, transit_label, c='k')
    plt.scatter(size, data[0].squeeze().detach().cpu(), c='b')
    plt.scatter(size, reconstruction[0].squeeze().detach().cpu(), c='g')
    plt.show()


def shape_fit_performance(trained_model, test_loader, writer, iterations, device='cpu', plot=False, **kwargs):
    trained_model.eval()
    trained_model.to(device)
    total_kl_loss = 0.
    total_rec_loss = 0.
    batch_num = 0
    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            data, transit_model, _, _, anomaly_label = batch
            data = data.unsqueeze(dim=1).to(device)
            transit_model = transit_model.unsqueeze(dim=1).to(device)
            reconstruction, sampled, mean, kl_loss = model(data, extended_output=True)
            total_rec_loss = nn.functional.mse_loss(reconstruction, transit_model).item()
            total_kl_loss += kl_loss.item()
            batch_num += 1
            if plot and b % 10 == 0:
                plt.scatter(range(256), data[0].squeeze().cpu())
                plt.scatter(range(256), reconstruction[0].squeeze().detach().cpu(), c='g')
                plt.show()
        total_kl_loss /= batch_num
        total_rec_loss /= batch_num
        writer.add_scalar("Rec_Loss/test", total_rec_loss, iterations)
        writer.add_scalar("KL_Loss/test", total_kl_loss, iterations)
    print(f"Testset performance: \n Rec: {total_rec_loss}, KL: {total_kl_loss}")
    return total_rec_loss


def loss_transit_fitting(model, batch, labels, writer, iterations, gamma, beta=1., verbose=False, fit_weight=64,
                         use_h_labels=False):

    rec, rec_fitted, sampled, mean, kl_loss, fitting_loss = model(batch, labels[-1], labels[-2],
                                                                  use_h_labels=use_h_labels)

    raw_loss = nn.functional.mse_loss(rec.squeeze(dim=1), labels[0])
    writer.add_scalar("Rec_Loss/train", raw_loss.item(), iterations)
    horizontal_scale_loss = fitting_loss[0]
    writer.add_scalar("Horizontal_Scale_Loss/train", horizontal_scale_loss.item(), iterations)
    horizontal_shift_loss = fitting_loss[1]
    writer.add_scalar("Horizontal_Shift_Loss/train", horizontal_shift_loss.item(), iterations)
    vertical_loss = fitting_loss[2]
    writer.add_scalar("Vertical_Fit_Loss/train", vertical_loss.item(), iterations)
    if use_h_labels:
        fitting_loss = vertical_loss
    else:
        fitting_loss = 2 * horizontal_scale_loss + 2 * horizontal_shift_loss + vertical_loss
    writer.add_scalar("Fit_Loss/train", fitting_loss.item(), iterations)

    # monotonic decreasing reconstruction weight
    mse_estimata = gamma**2
    gamma = np.sqrt(min(mse_estimata, raw_loss.detach().item()))
    rec_loss = raw_loss / (2 * gamma ** 2)

    loss = rec_loss + beta * kl_loss + fit_weight * fitting_loss
    writer.add_scalar("KL_Loss/train", kl_loss.item(), iterations)
    writer.add_scalar("VAE_Loss/train", (rec_loss + beta * kl_loss).item(), iterations)
    writer.add_scalar("Gamma/train", gamma, iterations)
    writer.add_scalar("Loss/train", loss.item(), iterations)

    if verbose:
        print(f"Iteration {iterations}, Loss: {loss.item()}, Rec: {raw_loss.item()}, Gamma: {gamma}, "
              f"KL: {kl_loss.item()}, H_scale: {horizontal_scale_loss.item()}, H_shift: {horizontal_shift_loss.item()},"
              f" Vertical: {vertical_loss.item()}")

    return loss, gamma, {'rec':rec, 'z_sampled':sampled, 'z_mean':mean, 'rec_fitted':rec_fitted,
                         'transit_model':labels[0], 'scale_label':3./labels[-1], 'shift_label':labels[-2]*batch.size()[-1]/2.}


def plot_transit_fitting(data, **kwargs):
    reconstruction = kwargs['rec']
    rec_transform = kwargs['rec_fitted']
    transit_model = kwargs['transit_model']
    scale_label = kwargs['scale_label']
    shift_label = kwargs['shift_label']
    size = range(len(data[0][0]))
    transit_label = transit_model[0].cpu()
    plt.plot(size, transit_label, c='k')
    plt.plot(size, horizontal_scale_n_shift(transit_label.unsqueeze(dim=0).unsqueeze(dim=0),
                                            scale_label[0].unsqueeze(dim=0), shift_label[0].unsqueeze(dim=0)).squeeze(),
             c='m')
    plt.scatter(size, data[0].squeeze().detach().cpu(), c='b')
    plt.scatter(size, reconstruction[0].squeeze().detach().cpu(), c='r')
    plt.scatter(size, rec_transform[0].squeeze().detach().cpu(), c='g')
    plt.show()


def transit_fit_performance(trained_model, test_loader, writer, iterations, device='cpu', vae=None,
                            use_h_labels=False, plot=False):
    trained_model.eval()
    trained_model.to(device)
    total_kl_loss = 0.
    total_rec_loss = 0.
    total_h_scale_loss = 0.
    total_h_shift_loss = 0.
    total_vertical_loss = 0.
    batch_num = 0
    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            data = batch[0].unsqueeze(dim=1).to(device)
            labels = [label.to(device) for label in batch[1:]]

            rec, rec_fitted, sampled, mean, kl_loss, fitting_loss = trained_model(data, labels[-1], labels[-2],
                                                                                  use_h_labels=use_h_labels)
            total_kl_loss += kl_loss.item()
            total_rec_loss += nn.functional.mse_loss(rec.squeeze(dim=1), labels[0]).item()
            total_h_scale_loss += fitting_loss[0].item()
            total_h_shift_loss += fitting_loss[1].item()
            total_vertical_loss += fitting_loss[2].item()
            batch_num += 1

            if plot and b % 10 == 0:
                plt.plot(range(256), labels[0][0].cpu(), c='k')
                plt.scatter(range(256), rec[0].squeeze(dim=1).detach().cpu(), c='r')
                plt.scatter(range(256), data[0].squeeze(dim=1).cpu())
                plt.scatter(range(256), rec_fitted[0].squeeze(dim=1).detach().cpu(), c='g')
                plt.show()

        total_kl_loss /= batch_num
        total_rec_loss /= batch_num
        total_h_scale_loss /= batch_num
        total_h_shift_loss /= batch_num
        total_vertical_loss /= batch_num

        writer.add_scalar("Rec_Loss/test", total_rec_loss, iterations)
        writer.add_scalar("KL_Loss/test", total_kl_loss, iterations)
        writer.add_scalar("Horizontal_Scale_Loss/test", total_h_scale_loss, iterations)
        writer.add_scalar("Horizontal_Shift_Loss/test", total_h_shift_loss, iterations)
        writer.add_scalar("Vertical_Fit_Loss/test", total_vertical_loss, iterations)


    print(f"Testset performance: \n Rec: {total_rec_loss}, KL: {total_kl_loss}, H_scale: {total_h_scale_loss},"
          f" H_shift: {total_h_shift_loss}, Vertical: {total_vertical_loss}")

    return total_rec_loss


def loss_residual_fitting(model, batch, labels, writer, iterations, gamma, beta=1., verbose=False, **kwargs):

    reconstruction, sampled, mean, kl_loss = model(batch, extended_output=True)

    raw_loss = nn.functional.mse_loss(batch, reconstruction)

    # monotonic decreasing reconstruction weight
    mse_estimata = gamma ** 2
    gamma = np.sqrt(min(mse_estimata, raw_loss.detach().item()))
    rec_loss = raw_loss / (2 * gamma ** 2)

    loss = rec_loss + beta * kl_loss
    writer.add_scalar("Res_Rec_Loss/train", raw_loss.item(), iterations)
    writer.add_scalar("Res_KL_Loss/train", kl_loss.item(), iterations)
    writer.add_scalar("Gamma/train", gamma, iterations)
    writer.add_scalar("ResVAE_Loss/train", loss.item(), iterations)

    if verbose:
        print(f"Iteration {iterations}, Loss: {loss.item()}, Rec: {raw_loss.item()}, Gamma: {gamma},"
              f" KL: {kl_loss.item()}")

    return loss, gamma, {'rec':reconstruction, 'z_sampled':sampled, 'z_mean':mean}


def plot_residual_fitting(data, **kwargs):
    reconstruction = kwargs['rec']
    size = range(data.shape[-1])
    plt.scatter(size, data[0].squeeze(dim=1).detach().cpu(), c='b')
    plt.scatter(size, reconstruction[0].squeeze(dim=1).detach().cpu(), c='g')
    plt.show()


def res_fit_performance(trained_model, test_loader, writer, iterations, device='cpu', vae=None, plot=False, **kwargs):
    trained_model.eval()
    trained_model.to(device)
    total_kl_loss = 0.
    total_rec_loss = 0.
    batch_num = 0
    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            if vae is not None:
                batch = batch[1] if vae == 'res' else batch[0]
            batch = batch.unsqueeze(dim=1).to(device)
            reconstruction, sampled, mean, kl_loss = model(batch, extended_output=True)
            total_rec_loss += nn.functional.mse_loss(batch, reconstruction).item()
            total_kl_loss += kl_loss.item()
            batch_num += 1
            if plot and b % 10 == 0:
                plt.scatter(range(256), batch[0].squeeze().cpu())
                plt.scatter(range(256), reconstruction[0].squeeze().detach().cpu(), c='g')
                plt.show()
        total_kl_loss /= batch_num
        total_rec_loss /= batch_num
        writer.add_scalar("Rec_Loss/test", total_rec_loss, iterations)
        writer.add_scalar("KL_Loss/test", total_kl_loss, iterations)
    print(f"Testset performance: \n Rec: {total_rec_loss}, KL: {total_kl_loss}")
    return total_rec_loss


def gather_residuals(trained_model, dataloader, plot=False, shuffle=True, batch_size=256, use_h_labels=False):
    dataset = []
    trained_model.to('cpu')
    with torch.no_grad():
        trained_model.eval()
        for batch in dataloader:
            data = batch[0].unsqueeze(dim=1)
            if use_h_labels:
                scale = batch[-1].to('cpu')
                shift = batch[-2].to('cpu')
                reconstruction, transit_model, _, _, _, _ = trained_model(data, scale, shift, use_h_labels=True,
                                                                          sample=False)
            else:
                reconstruction, transit_model, _, _, _, _ = trained_model(data, use_h_labels=False, sample=False)
            residuals = (data - transit_model).squeeze(dim=1).detach()
            dataset.append(residuals)
            if plot:
                size = range(len(batch[0][0]))
                plt.scatter(size, batch[0][0])
                plt.plot(size, transit_model[0].squeeze(dim=0), c='r')
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.show()
                plt.scatter(size, residuals[0].squeeze(dim=0))
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.show()
    dataset = torch.cat(dataset)
    dataset = min_max_norm_vectorized(dataset)*2.-1.
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def gather_residuals_centered(trained_model, dataloader, plot=False, shuffle=True, batch_size=256, return_labels=False):
    dataset = []
    labels = []
    trained_model.to('cpu')
    with torch.no_grad():
        trained_model.eval()
        for batch in dataloader:
            data = batch[0].unsqueeze(dim=1)
            feature_model = batch[2]
            h_scale_correction = batch[3]
            reconstruction = trained_model(data, sample=False)

            reconstruction_fitted = reconstruction.detach()
            size = reconstruction_fitted.size()[-1]
            mid = size // 2
            # make h_scale consistent with feature model
            for s, h_scale in enumerate(h_scale_correction):
                new_size = (h_scale * size).round().int().item()
                if new_size == size:
                    continue

                resized = torch.nn.functional.interpolate(reconstruction_fitted[s].unsqueeze(dim=0), new_size, mode='linear',
                                                          align_corners=False,)
                if new_size > size:
                    mid_new = new_size // 2
                    start = mid_new - mid
                    reconstruction_fitted[s, :, :] = resized[:, :, start:start + size]
                else:
                    left_size = new_size // 2
                    right_size = new_size - left_size
                    reconstruction_fitted[s, :, mid - left_size:mid + right_size] = resized

            # vertical fit according to feature model
            maxi = reconstruction_fitted.max(dim=-1).values
            mini = reconstruction_fitted.min(dim=-1).values
            maxi_f = feature_model.max(dim=-1).values
            mini_f = feature_model.min(dim=-1).values
            reconstruction_fitted -= maxi.unsqueeze(dim=1)
            scale = (maxi_f-mini_f)/(maxi-mini).squeeze()
            reconstruction_fitted *= scale.unsqueeze(dim=1).unsqueeze(dim=1)
            reconstruction_fitted += maxi_f.unsqueeze(dim=1).unsqueeze(dim=1)
            residuals = (data - reconstruction_fitted).squeeze(dim=1)
            dataset.append(residuals)
            labels.append(batch[-1])
            if plot:
                size = range(len(data[0].squeeze()))
                plt.scatter(size, data[0].squeeze())
                plt.plot(size, batch[1][0].squeeze(dim=0), c='r')
                plt.scatter(size, reconstruction_fitted[0].squeeze(dim=0), c='g')
                plt.show()
                plt.scatter(size, residuals[0].squeeze(dim=0))
                plt.show()
    dataset = torch.cat(dataset)
    labels = torch.cat(labels)
    dataset = min_max_norm_vectorized(dataset)*2.-1.
    if return_labels:
        dataset = torch.utils.data.TensorDataset(dataset, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def training_loop(model, trainloader, loss_function, plot_function=lambda x, **kwargs: None, beta=1.,
                  testloader=None, num_epochs=200, lr=5e-4, print_every=20, plot_every=300, test_every=1000, vae=None,
                  use_h_labels=False, centered=False, device='cuda:0', save_path=None):

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gamma = 1.0  # float("Inf")
    writer = SummaryWriter()
    best_test_result = np.inf
    best_epoch = 0
    best_model = None
    iterations = 0
    start = time()
    for epoch in range(1, num_epochs+1):
        epoch_start = time()
        epoch_loss = 0.
        num_batches = 0
        for batch in trainloader:

            if isinstance(model, TransitShapeVAE) or centered:
                data = batch[0]
                labels = [label.to(device) for label in batch[1:]]
            elif vae is not None:
                data = batch[1] if vae == 'res' else batch[0]
                labels = None
            else:
                data = batch
                labels = None

            data = data.unsqueeze(dim=1).to(device)

            loss, gamma, model_out = loss_function(model, data, labels, writer, iterations, gamma, beta,
                                                   use_h_labels=use_h_labels, verbose=iterations % print_every == 0)

            if iterations % plot_every == 0:
                plot_function(data, **model_out)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            num_batches += 1
            epoch_loss += (loss.item() - epoch_loss) / num_batches
            iterations += 1

            if testloader is not None and iterations % test_every == 0:
                if isinstance(model, TransitShapeVAE):
                    test_result = transit_fit_performance(model, testloader, writer, iterations, vae=vae, device=device,
                                                          use_h_labels=use_h_labels)
                elif centered:
                    test_result = shape_fit_performance(model, testloader, writer, iterations, device=device)
                else:
                    test_result = res_fit_performance(model, testloader, writer, iterations, vae=vae, device=device)
                model.train()
                if best_test_result > test_result:
                    best_test_result = test_result
                    best_epoch = epoch
                    best_model = deepcopy(model)

        print(f"Average Loss Epoch {epoch}: {epoch_loss}, Duration: {time()-epoch_start} sec")
        writer.flush()
    writer.close()
    print(f"Training for {num_epochs} completed in {(time()-start)/60.} minutes.")

    model = model.to('cpu')
    best_model = best_model.to('cpu')
    if save_path is not None:
        pickle.dump(model, open(save_path + ".pkl", "wb"))
        pickle.dump(best_model, open(save_path + f"_best{best_epoch}.pkl", "wb"))

    return model


def train_model(trainloader, config, mean_transit_model, centered=False, version="ResidualVAE", validation_loader=None,
                transit_vae=None, num_epochs=100, test_every=250, plot_every=500, beta=.1, save_path=None):
    if version == "VAE":
        model = CustomVAE(config)
        return training_loop(model, trainloader, loss_residual_fitting, num_epochs=num_epochs, test_every=test_every,
                             plot_every=plot_every, plot_function=plot_residual_fitting, testloader=validation_loader,
                             beta=beta, vae='vae', save_path=save_path)
    elif version == "TransitVAE":
        model = TransitShapeVAE(config, mean_transit_model)
        if centered:
            return training_loop(model, trainloader, loss_shape_fitting, num_epochs=num_epochs, test_every=test_every,
                                 plot_every=plot_every, plot_function=plot_shape_fitting, testloader=validation_loader,
                                 beta=beta, centered=True, save_path=save_path)
        else:
            return training_loop(model, trainloader, loss_transit_fitting, num_epochs=num_epochs, test_every=test_every,
                                 plot_every=plot_every, plot_function=plot_transit_fitting, beta=beta,
                                 testloader=validation_loader, save_path=save_path)

    elif version == "ResidualVAE":
        assert transit_vae is not None, "Need instance of TransitVAE to train ResidualVAE! Please provide."
        if centered:
            restrainloader = gather_residuals_centered(transit_vae, trainloader)
            if validation_loader is not None:
                resvalloader = gather_residuals_centered(transit_vae, validation_loader)
            else:
                resvalloader = None
        else:
            restrainloader = gather_residuals(transit_vae, trainloader)
            if validation_loader is not None:
                resvalloader = gather_residuals(transit_vae, validation_loader)
            else:
                resvalloader = None
        model = CustomVAE(config)
        return training_loop(model, restrainloader, loss_residual_fitting, num_epochs=num_epochs, test_every=test_every,
                             plot_every=plot_every, plot_function=plot_residual_fitting, testloader=resvalloader,
                             beta=beta, save_path=save_path)
    else:
        raise RuntimeError("Unknown model version! Pick one from: [VAE, VAE-large, TransitVAE, ResidualVAE]")


if __name__ == "__main__":
    # load trainset and split in train and validation set
    dataset = pickle.load(open("data/alt_h", "rb"))

    # reformat ground truth labels
    shift = dataset.tensors[-2]
    shift /= dataset.tensors[0].size()[-1] / 2.
    scale = 3. / dataset.tensors[-1]
    dataset.tensors = (*dataset.tensors[:-2], shift, scale)

    validation_size = int(0.1 * len(dataset))
    trainset, valset = torch.utils.data.random_split(dataset, [len(dataset) - validation_size, validation_size],
                                                     generator=torch.Generator().manual_seed(42))
    trainloader = torch.utils.data.DataLoader(trainset.dataset, batch_size=256, shuffle=True, drop_last=False)
    valloader = torch.utils.data.DataLoader(valset.dataset, batch_size=256, shuffle=False, drop_last=False)

    theoretical_models = trainloader.dataset.tensors[0]
    mean_transit_model = theoretical_models.mean(dim=0)

    # diagnostc plot of mean transit model and variance
    plt.plot(range(theoretical_models.size()[1]), mean_transit_model)
    plt.show()
    print("Variance: {}".format(((theoretical_models - theoretical_models.mean(dim=0)) ** 2).mean()))

    # load config
    config = json.load(open("configs/vae_config.json", "r"))
    # For VAE-large
    # config = json.load(open("configs/vae_large_config.json", "r"))

    train_model(trainloader, config, mean_transit_model, centered=False, version="ResidualVAE",
                validation_loader=valloader, transit_vae=pickle.load(open("model_checkpoints/alt_h/transit_vae", "rb")),
                num_epochs=100, test_every=250, plot_every=500, beta=.1,
                save_path="model_checkpoints/alt_h/residual_vae")
