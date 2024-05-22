# Training/validation/testing data split protocol used for each benchmark forecasting dataset
# Following the same protocol for testing subset is important to make a fair comparison as with even a small 
# difference in the testing subset, the final performance can be considerably different.  
# This split protocol has been widely accepted in many forecasting models (e.g., "Are Transformers Effective for Time Series Forecasting?"  
# https://github.com/cure-lab/LTSF-Linear/blob/main/pics/Mul-results.png )
LIST_OF_FORCASTING_BENCHMARK_LENGTH = [    
    {   
    "name": "ETTh",
    "training_length": [0, 12 * 30 * 24],
    "val_length": [12 * 30 * 24 , 12 * 30 * 24 + 4 * 30 * 24],
    "testing_length": [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    },
    {   
    "name": "ETTm",
    "training_length":[0, 12 * 30 * 24 * 4],
    "val_length": [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4], # for the start val step, seq length (look back) should be additionally taken away 
    "testing_length": [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 
                       12 * 30 * 24 * 4 + 8 * 30 * 24 * 4] # for the start testing step, seq length (look back) should be additionally taken away 
    }
]