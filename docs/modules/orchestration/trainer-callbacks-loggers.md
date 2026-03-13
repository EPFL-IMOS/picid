# Trainer, Callbacks, and Loggers

The trainer is Hydra-instantiated and receives:

- callbacks (including checkpoint wrappers)
- configured loggers (WandB/TensorBoard/etc.)
- model and datamodule

Callbacks and loggers are the communication backbone for monitoring, checkpoints, and reporting.
