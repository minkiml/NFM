
from Classification.data_factory.data_factory_scla import classification_dataset

def Load_dataset(
                 argw):
    data_ = classification_dataset(
                                path = argw["data_path"],
                                sub_dataset = argw["sub_dataset"],

                                sr_train = argw["sr_train"],
                                sr_test = argw["sr_test"],
                                mfcc = argw["mfcc"],
                                dropped_rate = argw["drop_rate"],

                                batch_training = argw["batch_training"],
                                batch_testing = argw["batch_testing"],
                                num_workers = 10)
    return data_