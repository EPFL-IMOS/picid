# Git contributor statistics (scoped paths)

Statistics are derived from this repository’s Git history: branch **`dev`** only (**merge commits excluded**).

## Scope

Commits reachable from **`dev`** that modify at least one tracked path under any of:

- `configs/`
- `lmetk/`
- `picid/`
- `test/`
- `scripts/`
- `picid_report/`
- `picid_report_dashboard/`

Work that exists only on other branches never merged into **`dev`** is not included.

## Definitions

| Term | Meaning |
|------|---------|
| **Lines added / deleted** | Per-file insertions/deletions from `git log --numstat`, summed across file lines touched in commits within scope. |
| **Lines edited** | Lines added + lines deleted (gross churn, not LOC ownership). |
| **Net** | Lines added − lines deleted. |
| **Commit share** | Author’s commits ÷ total commits in scope. |
| **Net share** | Author’s net ÷ total net in scope (`236,096` below; authors with negative net would show a negative share). |

Distinct author strings are counted separately (for example **`ludocomito`** vs **`Ludovico Comito`** would appear as separate rows if both had commits on **`dev`** in scope).

---

## Contributors (branch `dev`)

| Author | Commits | Commit share % | Lines added | Lines deleted | Lines edited | Net | Net share % |
|--------|--------:|---------------:|------------:|--------------:|-------------:|----:|-----------:|
| Lev Telyatnikov | 786 | 52.47 | 466,014 | 268,450 | 734,464 | 197,564 | 83.68 |
| Raffael Theiler | 544 | 36.32 | 89,008 | 63,445 | 152,453 | 25,563 | 10.83 |
| vole | 116 | 7.74 | 5,694 | 1,880 | 7,574 | 3,814 | 1.62 |
| jaryP | 47 | 3.14 | 7,668 | 3,134 | 10,802 | 4,534 | 1.92 |
| arthur-bizzi | 5 | 0.33 | 4,707 | 86 | 4,793 | 4,621 | 1.96 |
| **Total** | **1,498** | **100.00** | **573,091** | **336,995** | **910,086** | **236,096** | **100.00** |

---

## Regenerating

From the repo root:

```bash
git log dev --no-merges --format='%aN' -- \
  configs/ lmetk/ picid/ test/ scripts/ picid_report/ picid_report_dashboard/
git log dev --no-merges --numstat --format='%aN' -- \
  configs/ lmetk/ picid/ test/ scripts/ picid_report/ picid_report_dashboard/
```

Aggregate `numstat` lines (skip binary `-  -`) by the preceding `%aN` line, compute shares as in the table above.
