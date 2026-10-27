---
name: eg-new-feature
description: Build a new feature using the elephant/goldfish workflow — design doc, goldfish design check, implement, review, validate.
---

Build a new feature using the elephant/goldfish workflow. The aim: design before code, let a fresh goldfish stress-test the design doc, then implement, review, and validate.

If the user gave a GitHub issue URL or `#<number>`, fetch it with `run_command` using `gh issue view <number> --json title,body,labels,comments` and seed the design doc from it.

For atomistic simulations, structure optimizations, or ML potential training workflows, verify domain compatibility with `janus-core` and `aiida-mlip` capabilities before designing.

## Step 0: Confirm scope before designing

Restate the request back to the user in 1-2 sentences ("I read this as: <X>. Confirm or correct.").

Verify alignment with atomistic workflow training goals and required calculation dependencies.

Check affected calculators, scripts, pipelines, workflows, and test files.

## Step 1: Write the design doc

Print the design doc to the user. For substantial features, propose writing it to `docs/specs/<feature-name>-design.md` using `write_file` and ask the user before creating the file.

Required sections:

```
DESIGN DOC
- Why: <user problem this solves; cite GH issue, conversation, or PRD section>
- Scope: <what is in; what is explicitly out>
- Surfaces touched: <files / routes / models / controllers / components / DB tables / external services>
- Interfaces: <component props, function signatures, API request/response shapes, DB columns, queue arguments>
- UX flow: <click-by-click for UI; request-by-request for backend>
- MLIP Calculator / Potential: <e.g. MACE, SevenNet, CHGNet, or dummy test potential>
- Execution Environment: <micromamba janus env / headless flags (--no-tracker)>
- Provenance & Storage: <AiiDA nodes / local outputs / trajectory files>
- Failure modes: <what the user sees when each thing breaks; what the API returns>
- Verification criteria: <unit tests, E2E spec, manual steps>
- Out-of-scope follow-ups: <noted, not built>
```

If an `eg-prd` exists for this feature in `docs/prds/`, cross-reference requirements and success criteria.

For UI work, sketch the visual structure in plain text or pseudo-JSX.

### No-code gate

**Do NOT edit, write, scaffold, or refactor code until BOTH Pass B (Critic) AND Pass C (Readiness) in Step 2 close with their ready tokens (`design ready` + `implementation ready`).** Article rule, paraphrased: *"I do not want you to create code. We are not going to create code. Resist your impulse."* This holds until the design doc passes both gates — even if the user asks to skip ahead, even if the change "looks trivial", even if it is "just one line".

If the user explicitly asks to skip the gate ("just write the code", "skip the design doc", etc.), restate the gate, name the still-open passes, and require an explicit override ("yes, override the no-code gate") before any `write_file` or `edit_file` call. The exception is the `eg-fix-bug` skill for trivial fixes covered by its own Step 0 triviality gate.

The design doc itself, test names mentioned in chat (not yet on disk), and read-only exploration (`read_file`, `read_many_files`, `run_command` for `git status` / `git log` / `git diff`) are NOT code edits and are permitted.

## Step 2: Three-goldfish design check

Run the article's full design-stage protocol: three sequential `invoke_subagent` calls per round (or two on revisions — see below), each with no prior context. The combined gate is "ready iff critic AND readiness both sign off"; comprehension is informational.

Each pass uses `agent_name="generalist"` and gets ONLY the design doc (no chat history, no implementation intent, no other passes' output). The asymmetry is the value.

**Round 1 runs all three passes; round 2+ skips comprehension** (revisions are gap-driven, not structural — once the doc reads cleanly, it almost always still reads cleanly). On every round, run critic and readiness.

### Pass A — Comprehension (round 1 only)

Verifies the doc reads cleanly to a cold reader.

```
<<<COMPREHENSION_START>>>
You are a fresh reader with no prior context. Below is a design doc for a feature in the workflow-training (atomistic ML workflows and simulations using janus-core and aiida-mlip) repo. Do NOT critique it yet. Your job is to verify the doc reads clearly to someone who walks in cold.

Output two short sections in this order:

## What this feature does
2-5 sentences in your own words. The user-visible change. Who triggers it, when, what they get back.

## How the existing system works (per the doc)
2-5 sentences summarizing the current behavior the doc describes touching. Surfaces, controllers, components, message flow — whatever the doc references.

End your output with EXACTLY one of these closing lines, on its own line:
- comprehension passed       (the doc reads cleanly; no ambiguous sections)
- comprehension unclear      (one or more sections are too vague to paraphrase)

If you mark it unclear, list the ambiguous sections by heading before the closing line. Do NOT critique architecture choices here — that is the critic's job. Only flag things you genuinely cannot understand.

DESIGN DOC:

<PASTE FULL DESIGN DOC FROM STEP 1 HERE>
<<<COMPREHENSION_END>>>
```

### Pass B — Critic (every round)

Finds gaps that block implementation.

```
<<<DESIGN_START>>>
You are a fresh reviewer with no prior context. Below is a design doc for a feature in the workflow-training (atomistic ML workflows and simulations using janus-core and aiida-mlip) repo (AGENTS.md at the repo root has the architecture, or check legacy GEMINI.md if present).

Your job: read the design doc, then read the surfaces it claims to touch, and find holes BEFORE implementation starts. Specifically:

- Is the scope crisp?
- Are the interfaces concrete enough?
- Do the verification criteria actually verify the feature?
- Does the doc misunderstand any existing code? Look up the surfaces it claims to touch and check.
- Are there failure modes the doc missed?
- Does the design specify `--no-tracker` for headless execution?
- Is `FrechetCellFilter` used for unit cell relaxation?
- Are structure files pre-validated with `tools/structure_inspector.py`?
- Are environment commands using `/opt/micromamba/envs/janus/bin/python` or `uv run --no-project`?
- Are there project-specific gotchas the doc ignores? AGENTS.md (or legacy GEMINI.md) is your reference.

Cross-check against the PRD in `docs/prds/` if one was generated.

DESIGN DOC:

<PASTE FULL DESIGN DOC FROM STEP 1 HERE>

Output: numbered list of gaps, with file:line citations where applicable. End with `design ready` ONLY if you have zero gaps. Otherwise list them and end with `design needs revision`.
<<<DESIGN_END>>>
```

### Pass C — Readiness (every round)

Stricter than the critic: not "is the design good?" but "is the design _executable_ in one pass?"

```
<<<READINESS_START>>>
You are a fresh implementer with no prior context. Below is a design doc for a feature in the workflow-training (atomistic ML workflows and simulations using janus-core and aiida-mlip) repo. Imagine you've been told: "Implement this. First pass. No follow-up questions allowed." Could you?

For every interface, file path, function signature, API request/response shape, DB column, queue argument, component prop, message type, and verification criterion the doc claims, ask:
- Could I write the corresponding code without asking the author anything?
- Could I verify it works without asking what "works" means?
- Are the cited files and line numbers concrete enough that I'd open the right file and edit the right region?

Output a numbered list of EVERY question you would have to ask the author before you could ship. For each:
- The question itself, one sentence.
- The section of the doc that should have answered it but didn't.

If the list is empty, say "No open questions."

End with EXACTLY one of these closing lines, on its own line:
- implementation ready       (zero open questions; first-pass implementable)
- implementation not ready   (one or more open questions remain)

A design can be beautiful and still fail this gate. The critic asks "is the design good?"; you ask "is the design executable?".

DESIGN DOC:

<PASTE FULL DESIGN DOC FROM STEP 1 HERE>
<<<READINESS_END>>>
```

### Triage and loop

A round is **ready** iff Pass B closes with `design ready` AND Pass C closes with `implementation ready`. Comprehension is informational: log it, surface it to the user, but do not gate progress on it. If comprehension returns `comprehension unclear` AND the round is otherwise ready, still proceed — but flag in the final report that the doc was unclear in places.

If a round is **not ready**, bundle the critic gaps and readiness open questions (and comprehension feedback if it returned unclear) into a single revise prompt with section labels (`=== CRITIC GAPS ===`, `=== READINESS OPEN QUESTIONS ===`, `=== COMPREHENSION FEEDBACK ===`). Tell the elephant to address EVERY numbered gap across BOTH the CRITIC GAPS and READINESS OPEN QUESTIONS sections — do not collapse or skip because the numbering restarts. Each gap is either: addressed in a doc revision, or rebutted with a verbatim reason citing AGENTS.md (or GEMINI.md) or the user's words. Print the revised doc back to the user once both gates close.

Then re-run Pass B and Pass C against the revised doc (skip Pass A — see above). If the round still does not converge after **three revisions**, the feature is under-specified — **stop and ask the user** for more direction via `ask_question`.

## Step 3: Implementation plan

Once the design doc is `design ready`, write a short ordered implementation plan in chat:

1. Core atomistic calculation / model interface
2. Workflow script or orchestration logic
3. CLI / configuration interface
4. Unit and integration tests

For trivial features (single-component change), skip this step.

## Step 4: Implement

**Pre-flight check before any code edit:** confirm Step 2 closed both Pass B (Critic) AND Pass C (Readiness). If either is still open, the no-code gate from Step 1 still applies — return to Step 2 instead of proceeding.

Follow the plan using `edit_file`, `write_file`, and `run_command`. After each layer, briefly verify before moving on:

- Test calculator initialization with a small dummy or standard structure.
- Run single-step verification with `--no-tracker`.
- Validate full test suite passes via `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest`.

For CLI or pipeline tools, verify output log formatting and exit codes.

Verify calculation reproducibility, unit cell constraints, and error handling for convergence failures.

## Step 5: Hand off to review

Run the `eg-precommit-review` skill workflow explicitly if the user wants it, or carefully review your own diff.

## Step 6: Test gate

```sh
- `uv run --no-project --python /opt/micromamba/envs/janus/bin/python ruff check src tests`
- `uv run --no-project --python /opt/micromamba/envs/janus/bin/python pytest`
```

All required tiers must pass. For UI features, also do a final walkthrough of the golden path AND the most plausible edge case.

## Step 7: Final report

Print to the user:
- Feature summary (one line)
- Files touched
- Tests added
- Design-check result (gaps surfaced and how each was resolved)
- Test gate status
- Out-of-scope follow-ups noted in the design doc

**STOP.** Do NOT commit. Wait for the user's literal commit instruction. Do NOT commit without explicit user permission. Follow Conventional Commits format when authorized.