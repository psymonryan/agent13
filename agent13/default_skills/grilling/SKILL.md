---
name: grilling
description: Interview the user one question at a time to reach shared understanding before acting. Use when the user describes a plan, idea, or change and alignment is needed, or when the user says "grill me" or "interview me".
license: MIT
metadata:
  version: 1.0.0
  author: Agent13
  source: "https://github.com/mattpocock/skills (MIT, adapted)"
  source_path: skills/productivity/grilling/SKILL.md
  source_ref: "85f83d3 2026-08-20"
  created: 2026-08-22
  category: development
  tags:
    - interview
    - planning
    - alignment
---

# Grilling

Interview the user **one question at a time** to reach shared understanding before acting. The goal is a shared mental model: every decision the user is making is settled, nothing is silently assumed, and the user has explicitly confirmed before you proceed.

## The design tree

As you interview, maintain a mental **design tree**: every decision branches into the sub-decisions that hang off it. You don't need to draw it — but you do need to track which branches are **settled** (the user has answered) and which are **open** (still to be resolved).

A question is **askable** (on the frontier) when every decision it depends on is already settled. A question that depends on an open decision is **not yet askable** — it goes to a later round.

## The loop

1. **Pick the next question** from the frontier. If multiple questions are askable, pick the one that unblocks the most downstream decisions.
2. **Ask it** — one question, not a batch. Keep it short. The user is often on a phone.
3. **Give your recommended answer** under the question, prefixed with `➡️`. This gives the user a default to react to rather than a blank page. They can say "yes" or correct you in a word.
4. **Wait for the answer.** Do not ask the next question until you have the user's response to this one.
5. **Record the answer** in the design tree. Mark the branch settled. Recompute the frontier.
6. **Repeat** until the frontier is empty.

## Format

Each question looks like this:

```
❓ **<short title>**: <the question, one or two sentences max>

➡️ <your recommended answer, one or two sentences>
```

Keep the whole message under ~5 lines. The user is on a small screen. If the question needs context, give the minimum context needed to answer it, not a background essay.

## Facts are yours, decisions are the user's

- **Facts** (what's in the codebase, what a tool returns, what a config says) are **your job to find**. Look them up yourself — `read_file`, `command`, the knowledge graph. Never ask the user for something you could find out.
- **Decisions** (what to build, how to do it, what to skip) are **the user's job**. Put each one to them and wait. Do not decide for them.

If a frontier question needs a fact you don't have, look it up **before** asking the question. Don't ask "should we use X or Y?" when you haven't checked whether X or Y is already in the codebase.

## Termination

The interview is done when the **frontier is empty**: every branch of the design tree is settled, nothing is left silently assumed.

When the frontier is empty, present a **summary**:

```
## Summary

- <decision 1>: <answer>
- <decision 2>: <answer>
- ...

Does this match what you had in mind? I won't start until you confirm.
```

**Do not act on the plan until the user explicitly confirms.** "I think we're done" is not confirmation. The user must say yes.

## Rules

- **One question at a time.** Never batch. The user is on a phone.
- **Recommended answer on every question.** Even if you're unsure, give your best guess — it's easier to correct than to fill in from scratch.
- **No acting before confirmation.** The interview produces understanding, not work. You do not write code, create files, or make changes during the interview.
- **If the user says "just do it" or "stop asking"** — stop the interview, confirm your understanding in one line, and proceed. Don't fight the user's desire to move.
- **If the problem is simple** (one or two decisions, no branches), don't force the full tree. Ask the one or two questions, confirm, done. The structure is for when there's genuine complexity.
