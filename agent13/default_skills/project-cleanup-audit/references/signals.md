# Cleanup Signals

Each signal is a **lead**, not a finding. This file says what each one actually
tells you, and where it lies.

## Extension mix

```sh
git ls-files | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -20
```

Tells you the shape of the repo and where the mass is. A long tail of
one-or-two-file extensions is usually generated fixtures or committed build
output. Does **not** tell you anything is removable — a `.sql` count of 40 may
be 40 migrations, every one of them load-bearing.

## Legacy path signals

```sh
git ls-files | grep -E '(^|/)(archive|bak|backup|copy|dead|deprecated|legacy|old|obsolete|tmp|unused|wip)(/|\.|$)'
```

Finds paths that advertise their own staleness. High signal, cheap to check.

Where it lies: `old` and `tmp` appear in legitimate names (`threshold_old.py` is
a real thing in numerical code; `tmpfile` handling is real code). A hit is a
question — "is this still referenced?" — not an answer.

## Duplicate file names

```sh
git ls-files -z | xargs -0 -n1 basename | sort | uniq -d
```

Finds the same filename in several places. This is the signal behind most real
duplication: `utils.py` in three packages, `config.json` in two apps,
`README.md` copied per directory.

Where it lies: `__init__.py`, `index.ts`, `mod.rs`, `package.json` and
`SKILL.md` are *supposed* to repeat. Filter the language's structural filenames
out before reading the list.

## Exact duplicate files

```sh
git ls-files -z | xargs -0 md5 -r | sort > /tmp/cleanup-hashes.txt
cut -d' ' -f1 /tmp/cleanup-hashes.txt | uniq -d > /tmp/cleanup-dupes.txt
grep -Ff /tmp/cleanup-dupes.txt /tmp/cleanup-hashes.txt
```

Byte-identical content. The strongest evidence in the audit: if two files hash
the same, one of them is almost always redundant.

Where it lies — and this matters:

- **Two identical files can be two independent contracts.** A vendored copy that
  must not drift, a fixture per environment, a template per language. Merging
  them creates a coupling that did not exist.
- Check whether they change together before merging: `git log --oneline -- <a> <b>`.
  If they have never been edited in the same commit, they are probably
  independent.
- On Linux use `md5sum`; the output format differs (`hash  path` with two
  spaces), so `cut -d' ' -f1` still works but `grep -Ff` matching needs the
  same file.

## Duplicate function bodies

**This is the one signal the shell recipes cannot reproduce.** Upstream ships a
403-line parser that normalises whitespace, strips comments and hashes function
bodies across 16 languages. There is no honest one-liner equivalent.

Use the language's own detector:

| Language | Tool |
| --- | --- |
| Python | `pylint --disable=all --enable=duplicate-code <pkg>` |
| Java, C, C++, C#, JS, TS, Go, Ruby, PHP, Scala | PMD CPD (`pmd cpd --minimum-tokens 100 --dir <path>`) |
| JS/TS | `jscpd` — needs node, which is **not installed here** |
| Go | `golangci-lint` with `dupl` |

If none is available, say so in the report and fall back to reading the largest
files. A false negative here is acceptable; a fabricated duplicate list is not.

## TODO / FIXME clusters

```sh
git grep -n -E '\b(TODO|FIXME|HACK|XXX|deprecated|legacy|unused|obsolete|dead code)\b'
```

Clusters are the signal, not individual hits. One `TODO` is a note; forty in one
module is a module nobody finished. Count by file:

```sh
git grep -c -E '\b(TODO|FIXME|HACK|XXX)\b' | sort -t: -k2 -rn | head -20
```

Where it lies: `deprecated` is frequently a *correct* annotation on a live
API — it is documentation of intent, not evidence of rot. Read the line before
you classify it.

A zero result is a real result, but `git grep` exits **1** when nothing matches.
In a script with `set -e` that aborts the audit, so append `|| true` when
chaining. Confirm the zero is genuine before reporting it — a typo in the
pattern looks exactly like a clean repo.

## Largest files

```sh
git ls-files -z | xargs -0 ls -l | sort -k5 -rn | head -25
```

Finds committed binaries, datasets, build output and single-file monoliths.
High signal for "this should not be in source"; no signal at all for "this
should be deleted".

## Untracked and ignored clutter

```sh
find . -type f -not -path './.git/*' | wc -l
```

Compare against `git ls-files | wc -l`. A large gap means the working tree holds
material git does not track — scratch files, local databases, generated output.
That is workspace hygiene, not code cleanup, but it belongs in the report
because it is usually the fastest win.

## Unused exports and dependencies

Neither is detectable by grep alone. Use the project's tooling:

- JS/TS: `npx knip`, `npx depcheck` (needs node).
- Python: `vulture <pkg>`, `pip-audit` for the dependency half.
- Go: `staticcheck` reports unused code.
- Rust: `cargo udeps`, and the compiler's own `dead_code` lint.

Report which tool you ran. "No unused dependencies found" is only meaningful
next to the name of the tool that looked.

## Tool-owned paths

Some directories are not part of the project's organisation at all — their
location is fixed by a tool that reads them there. They are never findings, no
matter how much they look like clutter.

| Path | Owned by | Why it cannot move |
| --- | --- | --- |
| `.agent13/` | this agent | project-local `skills/` and `saves/`; discovery and context restore both look here |
| `.git/` | git | the repository itself |
| `.github/` | GitHub | workflows, issue templates, actions |
| `.claude/`, `.cursor/`, `.codex/` | other agents | their own project config |
| `.vscode/`, `.idea/` | editors | workspace settings, launch configs |
| `.devcontainer/` | dev containers | container definition |
| `.venv/`, `node_modules/`, `__pycache__/`, `.pytest_cache/` | toolchains | regenerable state |

Two different treatments, and the difference matters:

- **Never move, never delete** — `.agent13/`, `.git/`, `.github/`, editor and agent config. These are *part of the project*; their location is a contract with the tool that reads them.
- **Exclude from clutter counts** — `.agent13/saves/`, `.venv/`, `node_modules/`, `__pycache__/`. Regenerable state. Not a finding, and not worth reporting as untracked clutter.

A hidden dot-directory full of config is the most common false positive this
audit produces. The tell: it is named after a **tool**, not after the project's
domain. If the name would appear in a `--help` output, it is tool-owned, and the
audit's answer is "leave it".

## What none of these signals can tell you

- Whether a file is loaded dynamically, by string name, by config, by a route table, by a migration runner, or by an external consumer.
- Whether a public API is used outside this repo.
- Whether a fixture is read by CI.
- Whether a "dead" branch is dead on purpose.

For each of those, search explicitly and write down what you searched. The
negative result is the evidence:

```sh
git grep -n 'module_name\|ModuleName\|module-name'   # every spelling
git log --oneline -5 -- path/to/file                  # last touched when, by whom
```
