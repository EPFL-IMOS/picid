# Sequencers, Windowing, and Striding

Window extraction is delegated to sequencers in `picid.data.optimization.sequencer`.

## Components

- `SlidingWindowBatchDataset`: thin dataset wrapper around sequencers.
- `DenseArraySequencer`: regular arrays.
- `RaggedArraySequencer`: variable-length per-unit arrays.
- core indexing engine for autoregressive window selection.

## Key behavior

- stride controls overlap density
- units are sequenced independently
- invalid windows are filtered by length constraints
- optional warmup/padding behavior depends on sequencer settings

API: [picid.data.optimization](../../../reference/api/picid_data_optimization.md)
