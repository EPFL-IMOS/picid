# Hooks and Inverse Scaling

Hooks execute side effects during evaluation:

- save predictions
- generate plots
- push artifacts to logger backends

Scaling wrappers can apply inverse transforms before metric computation so metrics are reported in physical units when required.
