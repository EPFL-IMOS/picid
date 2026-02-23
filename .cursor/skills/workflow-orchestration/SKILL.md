---
name: workflow-orchestration
description: Orchestrates agent workflow with plan-first execution, subagent offloading, self-improvement via lessons.md, and strict verification. Use when starting non-trivial tasks (3+ steps or architectural decisions), when something goes wrong, before marking any task complete, when given bug reports, when the user provides corrections, or when running verification steps.
---

# Workflow Orchestration

Guides disciplined execution: plan mode by default, subagents for focus, lessons from corrections, and proof before done.

## When to Use This Skill

| Trigger | Action |
|---------|--------|
| Non-trivial task (3+ steps or architectural decisions) | Enter plan mode first |
| Something goes sideways | STOP and re-plan; don’t keep pushing |
| Before marking a task complete | Run verification; prove it works |
| Bug report or failing test | Fix autonomously; no hand-holding |
| User correction | Update `tasks/lessons.md` with the pattern |
| Research, exploration, parallel analysis | Offload to subagents |

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for any non-trivial task.
- If things go wrong: STOP and re-plan; don’t keep pushing.
- Use plan mode for verification as well as building.
- Write detailed specs upfront to reduce ambiguity.

### 2. Subagent Strategy

- Use subagents often to keep the main context focused.
- Offload research, exploration, and parallel analysis to subagents.
- For complex problems: throw more compute via subagents.
- One task per subagent for clear ownership.

### 3. Self-Improvement Loop

- After any user correction: update `tasks/lessons.md` with the pattern.
- Write rules that prevent the same mistake.
- Iterate until the mistake rate drops.
- Review lessons at session start when relevant.

### 4. Verification Before Done

- Do not mark a task complete without proving it works.
- When relevant: diff behavior between main and your changes.
- Ask: “Would a staff engineer approve this?”
- Run tests, check logs, demonstrate correctness.

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask “is there a more elegant way?”
- If a fix feels hacky: “Knowing everything I know now, implement the elegant solution.”
- Skip for simple, obvious fixes; avoid over-engineering.
- Challenge your own work before presenting it.

### 6. Autonomous Bug Fixing

- When given a bug report: fix it; don’t ask for hand-holding.
- Use logs, errors, failing tests – then resolve them.
- Avoid unnecessary context switching for the user.
- Fix failing CI tests without being told how.

## Task Management

- **One task file per orchestration**: Create `tasks/<task-name>-YYYY-MM-DD.md` for each distinct workstream. Do not mix workstreams in one file.
- **Plan first**: Write plan with checkable items in the task file.
- **Index**: Keep `tasks/todo.md` as an index of active tasks, linking to each task file.
- **Verify plan**: Check in before starting implementation.
- **Track progress**: Mark items complete as you go.
- **Explain changes**: High-level summary at each step.
- **Document results**: Add review section to the task file.
- **When goal is finished**: Move the task file to `tasks/archive/YYYY-MM-DD/<task_name>.md` and remove from `tasks/todo.md`.
- **Capture lessons**: Update `tasks/lessons.md` after corrections.

## Core Principles

- **Simplicity first**: Keep changes as simple as possible; touch minimal code.
- **No laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal impact**: Only change what’s necessary. Avoid introducing new bugs.
