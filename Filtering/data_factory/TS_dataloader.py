'''
This is to get a dataloader for each domain dataset (k number)

'''
from Filtering.data_factory.data_factory_simulatedVib import Synthetic_data

def Load_dataset(
                 argw):
    data_ = Synthetic_data(
                            path = argw["data_path"], 
                            sub_dataset = argw["sub_dataset"],
                            varset_train = argw["varset_train"],
                            max_modes = argw["max_modes"],
                            diversity = argw["diversity"],
                            num_modes = argw["num_modes"],
                            
                            type_ = argw["filter_type"],
                            num_class = argw["class_num_filter"],
                            batch_training = argw["batch_training"])

    return data_, data_.__get_info__()
  