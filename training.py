from normalizer import Normalizer
from denormalizer import Denormalizer, SumDenormalizer

from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader


def train_with_sum_denormalizer(
    normalizer: Normalizer,
    model: nn.Module,
    denormalizer: SumDenormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer_class,
    optimizer_params: dict,
    num_epochs: int,
) -> None:
    """
    We assume that our output function is in the form y = f(x)g(sigma) + h(mu)
    We will have three different optimizers, one for each function
    """
    optimizers = [
        optimizer_class(denormalizer.get_mus_params(), **optimizer_params),
        optimizer_class(model.parameters(), **optimizer_params),
        optimizer_class(denormalizer.get_vars_params(), **optimizer_params),
    ]

    for phase, optimizer in enumerate(optimizers):
        for epoch in range(num_epochs):
            epoch_loss = 0
            for ts_indices, ts_x, ts_y in train_dataloader:
                # training loop
                optimizer.zero_grad()

                norm_ts, mus, vars = normalizer.normalize(ts_indices, ts_x)
                proc_ts = model(norm_ts) if phase != 0 else Tensor(0)
                out = denormalizer(proc_ts, mus, vars, phase)

                #####
                # we have to reshape ts_y to be of shape (batch_size, -1)
                ts_y = ts_y.reshape((ts_y.shape[0], -1))
                #####

                loss = criterion(out, ts_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] of phase {phase}, Loss: {epoch_loss:.4f}"
            )
    return


def simple_train(
    normalizer: Normalizer,
    model: nn.Module,
    denormalizer: Denormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer_class,
    optimizer_params: dict,
    num_epochs: int,
) -> None:
    params = (
        list(normalizer.parameters())
        + list(model.parameters())
        + list(denormalizer.parameters())
    )
    optimizer = optimizer_class(params, **optimizer_params)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for ts_indices, ts_x, ts_y in train_dataloader:
            # training loop
            optimizer.zero_grad()

            norm_ts, mus, vars = normalizer.normalize(ts_indices, ts_x)
            proc_ts = model(norm_ts)
            out = denormalizer(proc_ts, mus, vars)

            #####
            # we have to reshape ts_y to be of shape (batch_size, -1)
            ts_y = ts_y.reshape((ts_y.shape[0], -1))
            #####

            loss = criterion(out, ts_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return


def train(
    normalizer: Normalizer,
    model: nn.Module,
    denormalizer: Denormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer_class,
    optimizer_params: dict,
    num_epochs: int,
) -> None:
    if isinstance(denormalizer, SumDenormalizer):
        # we must use three different optimizers
        optimizers = [
            optimizer_class(denormalizer.get_mus_params(), **optimizer_params),
            optimizer_class(model.parameters(), **optimizer_params),
            optimizer_class(denormalizer.get_vars_params(), **optimizer_params),
        ]
    else:
        # we can use one optimizer with all the parameters
        params = (
            list(normalizer.parameters())
            + list(model.parameters())
            + list(denormalizer.parameters())
        )
        optimizers = [optimizer_class(params, **optimizer_params)]

    for phase, optimizer in enumerate(optimizers):
        for epoch in range(num_epochs):
            epoch_loss = 0
            for ts_indices, ts_x, ts_y in train_dataloader:
                # training loop
                optimizer.zero_grad()

                # retrieve/compute mus and vars
                norm_ts, mus, vars = normalizer.normalize(ts_indices, ts_x)
                # compute processed ts (0 if we are in phase 0 with SumDenormalizer)
                if phase == 0 and isinstance(denormalizer, SumDenormalizer):
                    proc_ts = Tensor(0)
                else:
                    proc_ts = model(norm_ts)
                # compute output
                if isinstance(denormalizer, SumDenormalizer):
                    out = denormalizer(proc_ts, mus, vars, phase)
                else:
                    # no need for the phase in this case
                    out = denormalizer(proc_ts, mus, vars)

                #####
                # we have to reshape ts_y to be of shape (batch_size, -1)
                ts_y = ts_y.reshape((ts_y.shape[0], -1))
                #####

                loss = criterion(out, ts_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] of phase {phase}, Loss: {epoch_loss:.4f}"
            )
    return
