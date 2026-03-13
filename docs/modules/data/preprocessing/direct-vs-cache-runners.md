# Direct vs Cache Runners

Preprocessing execution is implemented with two runner paths:

- `DirectPipelineRunner`: no cache, full load->split->transform each run.
- `CachingPipelineRunner`: restores from final or boundary caches when valid.

Both execute via `PreProcessor` and use the same transform sequence protocol.
