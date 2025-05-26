import torch


def dl_to_sklearn__with_context_for_tabpfn(dl, lag=0):
    X, y = [], []
    for batch in iter(dl):
        x_c, y_c, x_t, y_t = batch
        X.append(torch.concat([x_t[:, lag, :], y_c.squeeze(2)], dim=1))
        y.append(y_t[:, lag, :])

    return torch.concat(X).numpy(), torch.concat(y).numpy()
