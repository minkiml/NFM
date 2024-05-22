import os
import logging
from torch.utils.data import DataLoader
from Classification.data_factory.speech_dataset.speech_commands import SpeechCommands

class classification_dataset(object):
    def __init__(
                # Loaded dataset
                self, 
                path = "",
                sub_dataset = "SpeechCommands",
                sr_train = 1,
                sr_test = 1,
                mfcc = 1,
                dropped_rate = 0,

                batch_training = 32,
                batch_testing = 32,
                num_workers = 0):
        # Global logger
        log_loc = os.environ.get("log_loc")
        root_dir = os.path.abspath(os.path.join(os.getcwd(), '..')) # go to one root above
        logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all'), level=logging.INFO,
                            format = '%(asctime)s - %(name)s - %(message)s')
        self.logger = logging.getLogger('From classification dataset')
        self.path = path
        self.sub_dataset = sub_dataset

        self.mfcc = mfcc
        self.drop_rate = dropped_rate
        self.sr_train = sr_train
        self.sr_test = sr_test

        self.num_workers = num_workers
        self.batch_training = batch_training
        self.batch_testing = batch_testing
        self.__read_and_construct__()
    def __read_and_construct__(self):
        # Check if the dataset already exists in the root directory

        dataset = {
            "SpeechCommands": SpeechCommands
        }[self.sub_dataset]

        training_set = dataset(
            partition="train",
            path = self.path,
            mfcc=self.mfcc,
            sr=self.sr_train,
            dropped_rate=self.drop_rate,
            batch_size=self.batch_training
        )
        test_set = dataset(
            partition="test",
            path = self.path,
            mfcc=self.mfcc,
            sr=self.sr_train
            if self.sr_test == 0
            else self.sr_test,  # Test set can be sample differently.
            dropped_rate=self.drop_rate,
            batch_size=self.batch_testing
        )
        if self.sub_dataset in [
                "SpeechCommands"
            ]:
            validation_set = dataset(
                partition="val",
                path = self.path,
                mfcc=self.mfcc,
                sr=self.sr_train,
                dropped_rate=self.drop_rate,
                batch_size=self.batch_training
            )
        else: validation_set = test_set


        if self.sub_dataset == "SpeechCommands" and self.mfcc == 0:
            test_set_sr2 = dataset(
            partition="test",
            path = self.path,
            mfcc=self.mfcc,
            sr=2,  # Test set can be sample differently.
            dropped_rate=self.drop_rate,
            batch_size=self.batch_testing
            )

            test_set_sr3 = dataset(
            partition="test",
            path = self.path,
            mfcc=self.mfcc,
            sr=4,  # Test set can be sample differently.
            dropped_rate=self.drop_rate,
            batch_size=self.batch_testing
            )
        else: test_set_sr2, test_set_sr3 = None, None
        # Create a DataLoader to efficiently load the dataset in batches
        self.training_loader = DataLoader(training_set, 
                                          batch_size=self.batch_training, 
                                          shuffle=True,
                                          num_workers= self.num_workers,
                                          drop_last=True)
        
        self.val_loader = DataLoader(validation_set, 
                                         batch_size=self.batch_testing, 
                                         shuffle=False,
                                         num_workers= self.num_workers,
                                         drop_last=False)
        
        self.testing_loader = DataLoader(test_set, 
                                         batch_size=self.batch_testing, 
                                         shuffle=False,
                                         num_workers= self.num_workers,
                                         drop_last=False)
        if self.sub_dataset == "SpeechCommands" and self.mfcc == 0:

            self.testing_loader2 = DataLoader(test_set_sr2, 
                                            batch_size=self.batch_testing, 
                                            shuffle=False,
                                            num_workers= self.num_workers,
                                            drop_last=False)
            self.testing_loader3 = DataLoader(test_set_sr3, 
                                            batch_size=self.batch_testing, 
                                            shuffle=False,
                                            num_workers= self.num_workers,
                                            drop_last=False)
        else: self.testing_loader2, self.testing_loader3 = None, None
        del training_set, validation_set, test_set, test_set_sr2, test_set_sr3
        self.logger.info(f"Training {self.sub_dataset} data loader is constructed. The total number of mini-batches: {len(self.training_loader)}")
        self.logger.info(f"Testing {self.sub_dataset} data loader is constructed. The total number of mini-batches: {len(self.testing_loader)}")
        self.logger.info(f"Validation {self.sub_dataset} data loader is constructed. The total number of mini-batches: {len(self.val_loader)}")

    def __get_training_loader__(self):
        return self.training_loader
    def __get_testing_loader__(self):
        return self.testing_loader
    def __get_val_loader__(self):
        return self.val_loader
    def __get_testing2_loader__(self):
        return self.testing_loader2
    def __get_testing3_loader__(self):
        return self.testing_loader3
    def __get_info__(self): # TODO
        return (self.L, self.C, self.num_class)