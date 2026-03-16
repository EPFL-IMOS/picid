# Intro

The interface is a helper class allowing the user to easily implements custom pipelines in the project. 
 
The interface class is EntryInterface (which you can find in the following python [file](../../picid/interface/interface.py)), and manges the training pipeline. 

It exposes three main functions: 

- `process_datasource(...)`: this function takes as input a datasource, both as string or AbstractDataSourceLoader for custom sources, a list of transforms functions, and returns the processed data source, ready to be used in the training loop of the interface. Internally, the function applies the transforms to the datasource data, and returns a transformed set of data.
- `train(...)`: this is the main entry point to the interface. It takes as input the datasource and a list of instantiated config files (see [schemas](schemas.md)), and lunch the training pipeline. See [`training logic`](#training-function-internal-logic) for more details about the internal training pipeline.
- `get_datasource()`: it takes as input a string representing the datasource, and returns the associated class.

To check a list of possible datasources, you can use the static function `get_available_datasources()`, which output is a list of strings. 


# Training function internal logic

Internally, picid uses Hydra config files to build an experiment. The function train() is an interface between the used and this aspect. To do it, internally, it convers all the input values into config files, which are then used to create the final config file, which is then run.   

Most of the config files expose a parameter called `model_class`, which points to the associated class within the module