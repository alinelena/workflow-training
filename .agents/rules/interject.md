# Rule: `/interject` — Dynamic Context Injection & First-Principles Rethink

Whenever the user sends a message starting with or containing `/interject` (e.g., `/interject <new information, constraint, or correction>`), you **MUST** treat it as a high-priority context pivot and follow this exact protocol:

## 1. Immediate Halt & Anti-Anchoring
- **Stop incremental patching**: Do NOT simply tack the new information onto your current plan or make a superficial band-aid edit to preserve work already done.
- **Drop sunk-cost bias**: Any code, design doc, or plan produced prior to `/interject` is disposable if the new information points to a cleaner or more accurate approach.

## 2. Assumption Audit
Explicitly review everything decided or assumed so far in the current task or workflow (including active `eg-*` stages, implementation plans, or debugging hypotheses):
- Identify which prior assumptions remain **valid**.
- Identify which prior assumptions, interfaces, or planned steps are **invalidated, contradicted, or rendered suboptimal** by the new information.

## 3. First-Principles Rethink
Re-evaluate the problem from scratch with the new information integrated as a primary constraint:
- Ask: *"If I had known this from turn 1, how would I have designed or solved this problem?"*
- If you are mid-workflow (e.g., inside `eg-brainstorm`, `eg-prd`, `eg-new-feature`, `eg-fix-bug`, or `eg-precommit-review`), rewind to the earliest workflow stage affected by the new information (such as updating the Seed, PRD, Problem Doc, or Design Doc) rather than forcing the old plan forward.

## 4. Structured Re-evaluation Output
Before making any further code edits or running mutating commands, print a concise **Interjection Re-evaluation** block in chat:

```markdown
### 🔄 Interjection Re-evaluation
- **New Information**: <1-2 sentence summary of the injected context/constraint>
- **Invalidated / Changed**: <bullet list of prior assumptions, files, or plan steps that no longer hold>
- **Rethought Approach**: <the revised design, root-cause hypothesis, or step-by-step plan from first principles>
- **Impact on In-Flight Work**: <what existing changes will be kept, reverted, or refactored>
```

## 5. Confirmation & Execution Gate
- **If the rethink introduces architectural trade-offs, scope changes, or multiple viable directions**: Use `ask_question` (or ask the user directly if trivial) to confirm the revised direction before modifying files.
- **If the new information is a clear, unambiguous directive**: Present the `Interjection Re-evaluation` block and immediately execute the revised approach.

