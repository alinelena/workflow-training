---
name: interject
description: >-
  Inject new information, constraints, or corrections mid-task via /interject
  and force the agent to audit prior assumptions and rethink its entire approach
  from first principles.
---

# `/interject` — Rethink Approach from New Information

When invoked via `/interject <new information>` (or when the user interjects with new context, constraints, or corrections), execute the following first-principles rethink workflow.

## Step 1: Parse the Interjection
If the user invoked `/interject` without providing the new information in the same message, ask them immediately:
> *"What new information, constraint, or correction would you like to inject?"*

Otherwise, extract the new information, constraint, or observation provided by the user.

## Step 2: Halt & Audit Current State
1. **Pause all in-flight implementation**: Do not write or edit code until the re-evaluation is complete.
2. **Inspect current state**:
   - Review the active conversation goal, current plan, or active workflow stage (e.g., `eg-brainstorm`, `eg-prd`, `eg-new-feature`, `eg-fix-bug`, `eg-precommit-review`).
   - If code has already been modified in the working tree, inspect `git status` and `git diff` to see what in-flight changes exist.
3. **Audit assumptions**: Separate prior assumptions into:
   - **Still Valid**: Constraints and facts that remain true.
   - **Invalidated / Superseded**: Decisions, architectures, or hypotheses that conflict with or are made obsolete by the new information.

## Step 3: Rethink from First Principles
Ask yourself:
> *"Given everything we know now—including this new information—what is the cleanest, most direct way to solve the user's underlying problem from scratch?"*

- Reject sunk-cost bias: do not preserve suboptimal code or design docs just because they were already written.
- If a fresh perspective or codebase re-check is needed, read the newly relevant files or spawn a fresh `research` / `self` subagent via `invoke_subagent` with the updated context.
- If an Elephant/Goldfish (`eg-*`) workflow is active:
  - Update the active artifact (Seed, PRD, Problem Doc, or Design Doc) to reflect the new reality.
  - Re-run the relevant Goldfish check if the design or diagnosis materially changed.

## Step 4: Present the Interjection Re-evaluation
Output the following structured summary to the user:

```markdown
### 🔄 Interjection Re-evaluation
- **New Information**: <concise restatement of what the user added>
- **Invalidated / Changed**:
  - <prior assumption or plan step 1 -> why it no longer holds>
- **Rethought Approach**:
  1. <new step 1>
  2. <new step 2>
- **Impact on In-Flight Work**: <what will be kept, refactored, or reverted>
```

## Step 5: Align and Proceed
- If the new information opens multiple architectural options or changes project scope, call `ask_question` to let the user choose the preferred path.
- If the revised path is unambiguous and directly actionable, proceed with executing the rethought approach.

