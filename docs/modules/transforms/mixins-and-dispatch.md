# Mixins and Dispatch

Transform fit/apply behavior across multi-unit data is controlled by mixins.

Common mixins:

- `ConcatFitAndPerSegmentTransformMixin`
- `NoFitPerSegmentMixin`
- `NoFitConcatAlongAxisMixin`
- `InverseTransformMixin`

Handlers dispatch dense/ragged combinations to the correct execution path.
