# Worktree And Environment Setup

This note records how the `worktrees/fix_tests` worktree was created from the
current `fix_tests` branch, how its Python environment was set up, and how
changes made inside that worktree can be brought back into the parent workspace.

## 1. Create the worktree

The parent workspace was already checked out on `fix_tests`, so the worktree was
created from the current branch tip in detached mode:

```bash
git worktree add --detach /Users/leone/projects/TabPHM/worktrees/fix_tests fix_tests
```

Detached HEAD is intentional here. Git does not allow the same branch to be
checked out as an active branch in two worktrees at once, and the parent
workspace already owns `fix_tests`.

## 2. Set up the environment

The working setup is to build the worktree environment with `uv`, not to copy
the parent `.venv`.

```bash
cd /Users/leone/projects/TabPHM/worktrees/fix_tests
uv sync --extra dev
```

Why `uv sync` is the preferred approach:

- A copied `.venv` can keep editable-package pointers to the original checkout.
- This repository uses local editable packages from `local_packages/`.
- `uv sync` rebuilds those editable installs against the worktree paths, so
  `picid` and the local packages resolve from the new worktree itself.

Because `local_packages/` is part of the repository checkout, it is already
present inside the worktree and is picked up automatically by `uv sync`.

## 3. Verify the environment

Two quick checks were used:

1. Verify imports resolve from the worktree:

```bash
.venv/bin/python -c "import picid; print(picid.__file__)"
```

Expected result:

```text
/Users/leone/projects/TabPHM/worktrees/fix_tests/picid/__init__.py
```

2. Run a lightweight pytest smoke check:

```bash
.venv/bin/python -m pytest -q \
  test/cli/test_validation.py \
  test/meta/test_collection_contract.py
```

This passed with:

```text
7 passed in 0.40s
```

## 4. Bring changes back to the parent branch

Since the worktree was detached, the change needs a temporary branch before it
can be merged back into the parent workspace branch.

Create a temporary branch in the worktree:

```bash
cd /Users/leone/projects/TabPHM/worktrees/fix_tests
git switch -c codex/worktree-setup-doc
```

Commit the worktree-local change:

```bash
git add WORKTREE_SETUP.md
git commit -m "docs: add worktree setup notes"
```

Then move to the parent workspace, which remains on `fix_tests`, and merge that
temporary branch:

```bash
cd /Users/leone/projects/TabPHM
git merge --no-ff codex/worktree-setup-doc
```

This is the handoff point from the child worktree back to the parent branch:

- the change is authored in the isolated worktree,
- committed on a temporary branch created from that worktree,
- then merged into the parent workspace branch (`fix_tests`).

## 5. Remove the temporary worktree after merge

Once the merge is complete and verified in the parent workspace, the temporary
worktree can be removed:

```bash
git worktree remove /Users/leone/projects/TabPHM/worktrees/fix_tests
git branch -d codex/worktree-setup-doc
```

This leaves the documentation change on the parent `fix_tests` branch while
cleaning up the temporary checkout used to prepare it.




---- Generalized workflow ----

Standard Operating Procedure: Git Worktree & Environment SetupThis document outlines the generalized process for creating isolated Git worktrees, configuring Python environments with uv, and reintegrating changes into a parent workspace. This workflow is designed for maximum isolation and is ideal for agentic workflows.1. Create the Worktree (Detached Mode)To work on a task without switching branches in your primary workspace, create a linked worktree. If the parent workspace already has the target branch checked out, you must use detached mode to avoid Git conflicts.Command:git worktree add --detach <target_path> <source_branch>
Why Detached?Git forbids having the same branch active in two separate worktrees at once. By using --detach, you check out the state of <source_branch> at its current commit without "occupying" the branch name itself, keeping it free for the parent workspace.2. Set Up the EnvironmentA worktree requires its own isolated virtual environment. In this repository, we use uv to handle dependencies and local editable packages.Workflow:# 1. Navigate to the new worktree
cd <target_path>

# 2. Synchronize the environment
uv sync --extra dev
Why uv sync?Path Integrity: A copied .venv contains absolute paths that point back to the original workspace.Editable Installs: If this project uses local packages (e.g., in local_packages/), uv sync ensures these are rebuilt against the current worktree's paths, preventing cross-workspace contamination.3. Verify the EnvironmentBefore performing automated tasks or agents start coding, verify that the Python interpreter is correctly resolving paths within the worktree.Verification Script:.venv/bin/python -c "import <main_package>; print(<main_package>.__file__)"
Expected: The output path should start with <target_path>.Smoke Test:.venv/bin/python -m pytest -q <path_to_smoke_test>
4. Bring Changes Back to the Parent BranchSince the worktree is in a detached state, you must create a temporary branch to "label" your commits before they can be merged.Inside the worktree, create a feature branch:git switch -c <task_branch_name>
Commit your changes:git add .
git commit -m "<type>: <description_of_changes>"
Return to the parent workspace and merge:cd <parent_workspace_path>
git merge --no-ff <task_branch_name>
5. Cleanup ProcedureAfter the merge is verified and successful in the parent workspace, remove the temporary artifacts.Remove the worktree directory:git worktree remove <target_path>
Delete the temporary local branch:git branch -d <task_branch_name>
Variable Reference TablePlaceholderDescription<target_path>The directory where the worktree will be created (e.g., ../worktrees/task-1).<source_branch>The base branch to checkout (e.g., main, develop).<main_package>The primary Python package name to verify imports.<task_branch_name>The temporary branch created to hold the work (e.g., feat/api-fix).
