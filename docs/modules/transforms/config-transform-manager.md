# ConfigTransformManager

`ConfigTransformManager` builds ordered transform programs from YAML and executes them split-wise.

Responsibilities:

- instantiate transforms from config
- preserve deterministic order
- expose transform metadata and inverter lookup
- provide cache-point aware transform subsets

See API: [picid.transforms.base](../../reference/api/picid_transforms_base.md)
