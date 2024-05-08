from Forecasting.data_factory.data_factory_fore import Forcasting_data
def Load_dataset(argw):
    data_ =  Forcasting_data(
                            path = argw["data_path"], 
                            sub_dataset = argw["sub_dataset"],
                            varset_train = argw["varset_train"],
                            varset_test = argw["varset_test"],
                            channel_independence= not argw["channel_dependence"],
                            # Forecasting setup
                            training_portion= argw["training_portion"], 
                            look_back= argw["look_back"],
                            horizon = argw["horizon"],

                            # loader params
                            batch_training = argw["batch_training"],
                            batch_testing = argw["batch_testing"],
                            num_workers = 10)
    return data_, data_.__get_info__()
