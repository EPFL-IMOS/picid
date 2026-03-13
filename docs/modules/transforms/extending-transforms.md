# Extending Transforms

1. Add transform class under `picid/transforms/`.
2. Choose compatible base class/mixins.
3. Implement `fit` only if stateful.
4. Add YAML entry with `_target_` + metadata.
5. Validate with train-only fit and split-wise behavior.

API:

- [picid.transforms](../../reference/api/picid_transforms.md)
- [picid.transforms.base](../../reference/api/picid_transforms_base.md)
