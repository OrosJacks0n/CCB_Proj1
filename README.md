# CCB_Proj1 — EE 5393 Homework #1

Python solutions for **EE 5393: Circuits, Computation, and Biology** (Winter 2026).

| File | Problem | Description |
|------|---------|-------------|
| `problem1a.py` | 1(a) | Gillespie SSA for outcome probabilities |
| `problem1b.py` | 1(b) | Mean / variance after 7 steps |
| `problem2_numba.py` | 2 | Lambda phage decision (Numba JIT, fast) |
| `problem3a.py` | 3(a) | CRN: Z = X · log₂(Y) |
| `problem3b.py` | 3(b) | CRN: Y = 2^(log₂(X)) |

See [`SOLUTIONS.md`](SOLUTIONS.md) for full write-ups and result tables.

**Requirements:** Python 3.12+. Problem 2 requires `numpy` and `numba`. Problem 3 requires `numpy`.

---

## Using this repository inside another repository

There are two standard Git approaches for embedding one repository inside another.

### Option A — Git Submodule (recommended)

A **submodule** keeps the two repositories independent: each has its own history, and the parent repo stores only a reference (commit SHA) to the child.

**Step 1 — Add this repo as a submodule of your parent repo**

```bash
# Run this from inside the parent repository
git submodule add https://github.com/OrosJacks0n/CCB_Proj1.git CCB_Proj1
git commit -m "Add CCB_Proj1 as submodule"
git push
```

The parent repo now has a `.gitmodules` file and a `CCB_Proj1/` directory that points to the specific commit you added.

**Step 2 — Clone the parent repo with its submodules**

```bash
git clone --recurse-submodules https://github.com/<you>/<parent-repo>.git
```

Or, if you already cloned the parent without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

**Step 3 — Update the submodule to a newer commit later**

```bash
cd CCB_Proj1
git pull origin main          # fetch the latest commit
cd ..
git add CCB_Proj1
git commit -m "Bump CCB_Proj1 submodule"
git push
```

---

### Option B — Git Subtree

A **subtree** merges the full history of this repo into a subdirectory of the parent. There is no separate `.gitmodules` file, so collaborators do not need to run any extra commands after cloning.

```bash
# Run this from inside the parent repository
git subtree add --prefix=CCB_Proj1 \
    https://github.com/OrosJacks0n/CCB_Proj1.git main --squash
git push
```

To pull in future updates:

```bash
git subtree pull --prefix=CCB_Proj1 \
    https://github.com/OrosJacks0n/CCB_Proj1.git main --squash
git push
```

---

### Which option should I choose?

| | Submodule | Subtree |
|---|---|---|
| Child history | Separate | Merged into parent |
| Extra clone step needed | Yes (`--recurse-submodules`) | No |
| Easy to contribute back | Yes | Harder |
| Complexity | Low for owners | Low for collaborators |

For most homework/project workflows where you just want the code available, **submodule** is the simpler choice.
