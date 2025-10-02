# OmegaConf Resolvers
We have implemented some functions to use in the cfgs.

- flat(path): flattens a dir, replaces / with +
- uuid(kind): generates a UUID
- sum(args): sum over all args
- quot(a,b): safe divides a//b (raises error when b=0).
- diff(a,b): a-b
- infer_data_dim(feature_name, dim): infers (fetches) the data dim after transforms. 