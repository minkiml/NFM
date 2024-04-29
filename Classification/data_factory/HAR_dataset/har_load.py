import os
import pathlib
import urllib.request
import tarfile
import torch
import torchaudio

from Classification.data_factory.speech_dataset.utils import normalise_data, split_data, load_data, save_data, subsample


class HAR(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: int,
        **kwargs,
    ):
        mfcc = kwargs["mfcc"]
        sr = kwargs["sr"]
        self.dropped_rate = kwargs["dropped_rate"]
        
        self.root = pathlib.Path(kwargs["path"])
        base_loc = self.root / "SpeechCommands" / "processed_data"
        X, y = 0, 0
        super(HAR, self).__init__(X, y)
    def _process_data(self):
        pass

        return (
            1
        )

    @staticmethod
    def load_data(data_loc, partition):

        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
            y = tensors["train_y"]
        elif partition == "val":
            X = tensors["val_X"]
            y = tensors["val_y"]
        elif partition == "test":
            X = tensors["test_X"]
            y = tensors["test_y"]
        else:
            raise NotImplementedError("the set {} is not implemented.".format(set))
        return X, y