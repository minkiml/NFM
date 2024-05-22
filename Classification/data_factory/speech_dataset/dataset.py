import torch
from Classification.data_factory.speech_dataset.speech_commands import SpeechCommands

from typing import Tuple


def dataset_constructor(
    config,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple (training_set, validation_set, test_set)
    """
    dataset = {
        "SpeechCommands": SpeechCommands
    }[config.dataset]

    eval_batch_size = config.batch_size

    training_set = dataset(
        partition="train",
        seq_length=config.seq_length,
        memory_size=config.memory_size,
        mfcc=config.mfcc,
        sr=config.sr_train,
        dropped_rate=config.drop_rate,
        valid_seq_len=config.valid_seq_len,
        batch_size=config.batch_size,
    )
    test_set = dataset(
        partition="test",
        seq_length=config.seq_length,
        memory_size=config.memory_size,
        mfcc=config.mfcc,
        sr=config.sr_train
        if config.sr_test == 0
        else config.sr_test,  # Test set can be sample differently.
        dropped_rate=config.drop_rate,
        valid_seq_len=config.valid_seq_len,
        batch_size=eval_batch_size,
    )
    if config.dataset in [
        "SpeechCommands"
    ]:
        validation_set = dataset(
            partition="val",
            seq_length=config.seq_length,
            memory_size=config.memory_size,
            mfcc=config.mfcc,
            sr=config.sr_train,
            dropped_rate=config.drop_rate,
            valid_seq_len=config.valid_seq_len,
            batch_size=eval_batch_size,
        )
    else:
        validation_set = None
    return training_set, validation_set, test_set


def get_dataset(
    config,
    num_workers: int = 4,
    data_root="./data",
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """
    training_set, validation_set, test_set = dataset_constructor(config)
    
    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if validation_set is not None:
        val_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        val_loader = test_loader

    # dataloaders = {"train": training_loader, "validation": val_loader}

    return training_loader, val_loader, test_loader