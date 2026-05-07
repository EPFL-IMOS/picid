# OmegaConf Resolvers

PICID uses **Hydra** and **OmegaConf** for configuration. Custom **resolvers** are registered in **`picid/run.py`** before `hydra.main` is invoked, so they are available in any YAML config loaded via the main entry point (e.g. `python picid/run.py ...`). In configs, use the syntax **`${resolver_name:arg1,arg2,...}`**. Resolvers are evaluated when OmegaConf resolves the config (e.g. when a value is first accessed); the two `infer_*` resolvers are replaced at runtime by the pipeline, so they only return real values after the pipeline has called the corresponding register function.

---

## Built-in resolvers (always available)

These are registered once at startup in `picid.run`.

| Resolver | Signature | Description |
|----------|-----------|-------------|
| **`flat`** | `flat(path)` | Replaces `/` with `+` in the string `path`. Useful for turning path-like keys into flat config keys. |
| **`uuid`** | `uuid(kind)` | Generates a UUID. `kind` can be `"short"` (default, 8-char hex) or `"hex"` (full hex). |
| **`sum`** | `sum(a, b, ...)` | Sum of all arguments. |
| **`prod`** | `prod(a, b, ...)` | Product of all arguments. |
| **`mod`** | `mod(x, y)` | Integer modulo: `int(x) % int(y)`. |
| **`int_div`** | `int_div(a, b)` | Integer division: `int(a) // int(b)`. |
| **`quot`** | `quot(a, b)` | Safe integer division: same as `a // b` but raises if `b == 0` or `a` is not divisible by `b`. If `b` is a list/tuple, tries each element until one divides `a`. |
| **`diff`** | `diff(a, b)` | `a - b` numerically; if either `a` or `b` is a string, returns `a` (no subtraction). |
| **`infer_data_dim`** | `infer_data_dim(key, dim)` | **Placeholder at startup:** returns the string `"not yet initialized"`. At runtime, the pipeline **replaces** this resolver with one that looks up the data array for `key` in the current data dictionary and returns the size of dimension `dim` (0-based). Used to wire config values (e.g. feature size) from actual data. |
| **`infer_dataloader_length`** | `infer_dataloader_length(key)` | **Placeholder at startup:** returns `"not yet initialized"`. At runtime, the pipeline **replaces** this resolver with one that looks up the dataloader length for `key` in a lengths dictionary. Used to set steps per epoch or similar from actual dataloaders. |

---

## Runtime-replaced resolvers

Two resolvers are registered first as placeholders and then **replaced** when the pipeline has the required data:

1. **`infer_data_dim`** — Replaced via **`register_data_dim_resolver(data)`** (in `picid.run`). The new implementation takes a `key` (string) and `dim` (int), looks up `data[key]`, and returns the size of that array along dimension `dim`. Supports NumPy arrays, PyTorch tensors, and Awkward arrays (and lists of them, if consistent).
2. **`infer_dataloader_length`** — Replaced via **`register_infer_dataloader_length_resolver(lengths)`** (in `picid.run`). The new implementation takes a `key` and returns `lengths[key]` (the dataloader length for that key).

If a config references these before they are replaced, the resolved value will be the placeholder string. Configs that depend on them should be evaluated after the pipeline has called the register functions.

---

## Where registration happens

All resolver registration is in **`picid/run.py`**:

- **Top-level (before `hydra.main`):** `flat`, `uuid`, `sum`, `mod`, `int_div`, `prod`, `quot`, `diff`, and the initial (placeholder) `infer_data_dim` and `infer_dataloader_length`.
- **During run setup:** The pipeline (or test harness) calls `register_data_dim_resolver(data)` and `register_infer_dataloader_length_resolver(lengths)` so that configs resolved later see the real dimensions and lengths.

---

## Example usage in YAML

```yaml
# Flatten a path for a config key
output_key: ${flat:experiment/railway/run}

# Unique run id
run_id: ${uuid:short}

# Arithmetic
channels: ${sum:4,8}
step: ${int_div:100,4}
remainder: ${mod:10,3}

# Safe division (errors if not divisible)
batch_per_epoch: ${quot:1000,32}

# Infer feature size from data (after resolver is replaced)
n_features: ${infer_data_dim:features,2}
```

---

## See also

- [Adding a new model](how_to_add_a_new_model.md) — How to wire model and experiment configs.
- [Setup](../getting-started/setup.md) — How to run experiments and override config from the CLI.

[← Guides index](index.md) | [Back to documentation index](../index.md)
