
# Model Types

The framework supports a variety of model types:

- SKLearn fit / predict
    - Goes trough the full dataset at once for each task
        - X (1 x N_samples x F)
        - y (1 x N_samples x F)

- Batchwise forward (contextual forward) 
    - Gives a windowed view for a specific time point 
        - target history (B x T_in  x F)
        - target futures (B x T_out x F)
        - context history (B x T_in  x F)
        - context futures (B x T_out x F)

- Full Batch ? The state of this is unclear
    - Gives a windowed view for a full dataset (careful, performance?)
        - target history (N_samples x T_in  x F)
        - target futures (N_samples x T_out x F)
        - context history (N_samples x T_in  x F)
        - context futures (N_samples x T_out x F)

What do Moiai, Chronosx and TimesFM need?
    - What de model wants to do in words is Take as much context as it can
        (from training to testing dataset) and then give exacatly 1 forecast for the next k timesteps.
        ```
        |----train––––|---test---|
              |---- context -----| -> |--forecast--|
        ```
    - X_train (N_samples x F)
    - X_test (N_samples x F)


But then, there is also fine-tuning on all of train data.

Problems:
    - We have 1 function for train one for test. But we need train and test data in test.
        - We could solve this in fit-predict, but then we need to remove the tabularizer from the pre-processing
    - We can not directly process the windowed batches. Some re-windowing is necessary.
    - We can most likely not use lightning to train these models, as they are heavily wrapped.
    - How to do regression with these models?


What do we want 
    - y_test (N_samples x horizon x F)


# How to impement?

- SKLearn fit / predict
    - Goes trough the full dataset at once for each task
        - X (1 x N_samples x F)
        - y (1 x N_samples x F)

        - load X_train_context (last part of training set) from cache
        - load y_train_context (last part of training set) from cache

        - the windowing could be implemented using `DenseArraySequencer` in the predict loop.

# What should happen during testing

## Rolling Context



step 1:
|-//---train––––|---test-----------------------------------------|
        |---- context -----| -> |--forecast--|


step 2:
move context window by xx
|-//---train––––|---test-----------------------------------------|
        xx|---- context -----| -> |--forecast--|


step 3:
move context window by xx
|-//---train––––|---test-----------------------------------------|
        xxxx|---- context -----| -> |--forecast--|



## Static Context
This is not fair.

|-//---train–––-----------–|---test-----------------------------------------|
        |---- context -----|---forecast-------------------------------------|







