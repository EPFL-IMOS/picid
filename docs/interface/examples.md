# Examples

### PHME20 with default configs 
In this example, we train an LSTM with 8 layers on `phme20` datasource. The task is defined by `Prognostic(task_type='rul')`. Once every component is defined using the corresponding config file, the training pipeline is started by calling the train() function.   

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                  metadata={'apply_to': 'rul', 'fit_on': 'train'})
    
    transforms = [scaler, scaler_target]
        
    model = LSTMConfig(n_layers=8)
    
    task_definition = Prognostic(task_type='rul')
    
    loggers = [CsvLogger(name='base_csv_logger', version='1')]
    
    callbacks = [EarlyStopping(monitor=None),
                 RichModelSummary()]
    
    evaluators = {s: RulEvaluatorConfig() for s in ['train', 'test', 'val']}
    
    interface = EntryInterface()
    interface.train(task_definition=task_definition, model=model, datasource='phme20', callbacks=callbacks,
                    loggers=loggers, evaluators=evaluators, transforms=transforms)

### PHME20 with default configs (Cont'd)

The same results can be achieved by getting the datalsource and operating on it before passing the resulting datasource to the training function

    scaler = DataTransform(transform_name='scaler_features', transform=MinMaxScalerSklearn(),
                           metadata={'apply_to': 'features', 'fit_on': 'train'})
    scaler_target = DataTransform(transform_name='scaler_targets', transform=MinMaxScalerSklearn(),
                  metadata={'apply_to': 'rul', 'fit_on': 'train'})
    
    transforms = [scaler, scaler_target]
    
    interface = EntryInterface()
    
    datasource = interface.get_datasource('phme20')
    processed_datasource = interface.process_datasource(datasource, transforms)
    
    ...
    
    interface.train(task_definition=task_definition, model=model, datasource=processed_datasource, callbacks=callbacks,
                    loggers=loggers, evaluators=evaluators)

## Custom datasource 

The functional interface gives the ability to use a custom datasource for training. The datasource can be composed of one or more sources. 

A single datasource CustomSingleSourceLoader can be instantiated manually by selecting the source type (Numpy array or DataFrame), or using the functions load_from_numpy and load_from_csv.
        
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "rul"],
    )
        
    datasource = CustomSingleSourceLoader.load_from_numpy(get_random_numpy_data(),
                                                  target_column=-1,
                                                  data_splitter=data_splitter,
                                                  task_mode='rul')

    interface = EntryInterface()
    
    processed_datasource = interface.process_datasource(datasource, transforms)

    interface.train(task_definition=task_definition, model=model, datasource=processed_datasource, callbacks=callbacks,
                    loggers=loggers, evaluators=evaluators)

A multi source datasource is just a collection of multiple separated sources. It can be created from a set of primitives (Numpy array or DataFrames) or from a set of already instantiated CustomSingleSourceLoader.
    
    data_splitter = TimeSplitter(
        train=0.5,
        val=0.25,
        test=None,
        seq_len=10,
        pred_len=5,
        create_splits_for=["features", "timestamps", "rul"],
    )
    
    sources = {str(i): get_random_numpy_data() for i in range(n_sources)}
    splitters = {str(i): deepcopy(data_splitter) for i in range(n_sources)}
    
    datasource = CustomMultiSourceLoader.load_from_primitive(sources,
                                                             target_column=-1,
                                                             data_name='test multi source',
                                                             task_mode='target',
                                                             data_splitter=splitters)
    
    interface = EntryInterface()
    
    processed_datasource = interface.process_datasource(datasource, transforms)
        
    ...
    
    interface.train(task_definition=task_definition, model=model, datasource=processed_datasource, callbacks=callbacks,
                    loggers=loggers, evaluators=evaluators)