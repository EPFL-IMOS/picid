# Schemas

To use the interface you need to specify a set of config files, one for each component of the pipeline. Such configs files are managed using pydantic, and are divided into ### categories: 

- Task definition ([task_definition.py](../../picid/interface/schemas/task_definition.py)): is the abstract definition that is extended by the specific ones. It contains the parameters of all possible tasks. By default, we implemented `Prognostic` and `Forecasting` configs.
- Models ([models folder](../../picid/interface/schemas/model/base.py)): a model config file contains all the parameters needed to use a specific model. Several models are present in the base code and ready to use, such as [lstm](../../picid/interface/schemas/model/lstm.py), [cnn1d](../../picid/interface/schemas/model/cnn1d.py), and [crossformer](../../picid/interface/schemas/model/crossformer.py)
- Evaluators ([evaluators.py](../../picid/interface/schemas/evaluators/evaluators.py)): 
- Loggers ([loggers.py](../../picid/interface/schemas/evaluators/evaluators.py)): 
