# PRD: RL-Guided Mutation Scheduling on the ARVO Dataset

## 0. How to read this document
This is a build specification for a student research project in a
Reinforcement Learning course. The deliverable is a small, honest
experiment, not a production fuzzer. When a choice is ambiguous, prefer
the simplest version that still produces measurable results, and leave a
clearly marked `# NOTE:` in the code rather than inventing behaviour.

**Goal in one sentence:** compare three strategies for choosing fuzzing
mutation operators — tabular Q-learning, a UCB1 bandit, and uniform random
— on real, already-fixed memory-safety bugs from the ARVO dataset, where
every crash is verified against the developer's fix.

This project only *rediscovers bugs that are already public and already
patched*, inside sealed Docker containers, to measure a learning
algorithm. It does not develop new exploits.

---

## 1. Background the implementer needs

- **Fuzzing:** repeatedly feed a program altered inputs and watch for
  crashes.
- **Mutation operator:** one small edit to an input (flip a bit, insert a
  byte, etc.). A fuzzer must repeatedly choose which operator to apply.
  This choice is what the agent learns.
- **Coverage:** how much of the program's code an input exercised.
  libFuzzer reports it. More coverage generally means more chance of
  reaching a bug.
- **ARVO:** a dataset of reproduced OSS-Fuzz vulnerabilities. Each case
  ships as two Docker images: `n132/arvo:<id>-vul` (crashes on the bug)
  and `n132/arvo:<id>-fix` (patched). Running `arvo` inside the container
  reproduces the bug; `arvo compile` recompiles.
- **AddressSanitizer (ASan):** compiled into the target; prints a line
  beginning `==ERROR: AddressSanitizer` when a memory bug triggers.

---

## 2. Scope

### In scope
1. A Python environment wrapper that drives one ARVO target.
2. Three agents: Random, UCB1, Q-learning.
3. An experiment runner (multiple cases, settings, repeats).
4. Metrics logging, crash verification, plots, and a results table.

### Out of scope (state as limitations in the report)
- No deep learning. The state space is tiny; a table is enough.
- No changes to the target programs or their harnesses.
- No new vulnerability discovery; only rediscovery of the known bug.
- No distributed or multi-core fuzzing. One target, one process at a time.

---

## 3. The six target cases
Start with 12950 only. Add the rest once the pipeline works.

| ID | Project | Input format | Crash type |
|----|---------|--------------|------------|
| 12950 | pcre2 | regex text | heap-buffer-overflow (read) |
| 13222 | file | any bytes | heap-buffer-overflow (read) |
| 3848 | yara | PE binary | heap-buffer-overflow (read) |
| 10304 | libxml2 | XML | use-after-free (read) |
| 3522 | zstd | raw bytes | heap-buffer-overflow (write) |
| 16091 | libspng | PNG | heap-buffer-overflow (read) |

Backups if an image fails to pull or run: 4088, 11752, 9847.

---

## 4. Architecture

```
run_experiment.py          # top-level: loops cases × agents × settings × seeds
  └── ArvoEnv              # wraps ONE target inside its container
        ├── reset()        # returns initial state, resets seed pool
        ├── step(action)   # applies operator, runs target, returns (state, reward, done, info)
        ├── _run_input()   # executes the fuzz binary on one input, parses coverage + crash
        └── _verify()      # re-runs a crash on the -fix image
  └── agents/
        ├── random_agent.py
        ├── ucb1_agent.py
        └── qlearning_agent.py
  └── mutators.py          # the 8 operators
  └── metrics.py           # logging, plots, stats
```

**Where the code runs:** the agent and environment run *inside* the
`-vul` container, with the project folder mounted via
`docker run -v $(pwd):/work`. Starting a container per input would be far
too slow. Launch the container once per run; do all executions inside it.

---

## 5. The MDP (what the agent learns)

### Action space (`mutators.py`)
Eight operators, each takes `bytes` and returns `bytes`:

| # | Operator | Effect |
|---|----------|--------|
| a1 | bit flip | flip one random bit |
| a2 | byte flip | XOR one random byte with 0xFF |
| a3 | random byte | replace one byte with a random value |
| a4 | insert | insert 1–4 random bytes at a random position |
| a5 | delete | delete 1–4 bytes at a random position |
| a6 | interesting value | overwrite 1/2/4 bytes with a boundary value from {0, 1, 0x7F, 0x80, 0xFF, 0x7FFF, 0xFFFF, 0x7FFFFFFF, 0xFFFFFFFF} |
| a7 | duplicate chunk | copy a 1–32 byte chunk to a random position |
| a8 | splice | join a prefix of the input with a suffix of another pool seed |

### State
`s = (previous_action, got_new_coverage, size_bucket)`
- `previous_action`: 0–8 (8 = "none yet")
- `got_new_coverage`: 0 or 1
- `size_bucket`: S (<64 B), M (64–1024 B), L (>1024 B)

Total states: 9 × 2 × 3 = **54**.

### Transition
1. Draw a seed uniformly at random from the pool.
2. Apply the chosen operator to produce `x'`.
3. Run the target on `x'`.
4. If `x'` reached new coverage, add it to the pool.

**Seed selection is uniform for ALL three agents.** Only operator choice
differs. This keeps the comparison fair.

### Reward
```
reward = min(new_coverage_features, 10) / 10   +   10 * crashed
```
Clipping stops one lucky input from dominating. The +10 makes a crash the
main goal.

### Episode
Ends at the first crash, or after `T = 5000` executions, whichever comes
first.

---

## 6. The three agents

All agents expose the same interface:
```python
class Agent:
    def select_action(self, state) -> int: ...
    def update(self, state, action, reward, next_state) -> None: ...
```

- **Random:** `select_action` returns a uniform choice from 0–7.
  `update` does nothing.
- **UCB1:** stateless bandit over the 8 operators. Try each once, then
  pick `argmax(mean_reward[a] + c * sqrt(ln(t) / n[a]))`, with `c = sqrt(2)`.
- **Q-learning:** table `Q[54][8]`, initialised to 0.
  - update: `Q[s,a] += alpha * (reward + gamma * max(Q[s',:]) - Q[s,a])`
  - `alpha = 0.1`, `gamma = 0.9`
  - ε-greedy: `epsilon = max(0.05, 1 - t / (0.2 * T))`

---

## 7. Environment details (`ArvoEnv`)

### Finding the fuzz binary
Inside the container, the fuzz target is usually under `/out/`. Find an
executable whose name contains `fuzz`. Confirm during feasibility and hard-code
the discovered path per case in a config dict.

### Running an input and reading coverage
Run the binary on a directory containing the single input, with
`-runs=0`, e.g.:
```
<fuzz_binary> -runs=0 /work/current_input_dir
```
Parse libFuzzer's stderr for `cov:` and `ft:` counts. "New coverage"
means the cumulative `ft:` (feature count) rose since the last best.

`# NOTE:` if a target does not print coverage counts, fall back to the
**crash-only reward** (drop the coverage term; reward is just `10 * crashed`).
Record in the output that coverage was unavailable for that case. Do not
fabricate coverage numbers.

### Detecting a crash
A crash occurred if the output contains `ERROR: AddressSanitizer` or the
process exits with the sanitizer's abort code. Save the crashing input.

### Verifying a crash (`_verify`)
A crash counts only if:
1. it reproduces on the `-vul` build, **and**
2. it does **not** crash on the `-fix` build.
Run the saved input through the `-fix` image. Keep the crash only if the
fixed build exits cleanly. This is the project's key correctness check.

### Deduplicating crashes
Two crashes are the same bug if the top three frames of their ASan stack
traces match. Keep one per group.

---

## 8. Experiment matrix

- **Setting A (from seeds):** start from the initial seed set. Realistic.
- **Setting B (perturbed PoC):** start from the PoC with 10% of its bytes
  randomly changed, re-checked so the perturbed file does *not* already
  crash. Controls difficulty so crash metrics are meaningful.

Runs: 3 agents × 6 cases × 2 settings × 5 repeats (random seeds 0–4)
= **180 runs**. Each run has budget `T = 5000`.

`# NOTE:` if execution is too slow, lower `T` and record the value used.
Report whatever you actually ran.

---

## 9. Metrics (`metrics.py`)

Per run, log to CSV: case, agent, setting, seed, executions_used,
first_crash_exec (or blank), verified (bool), final_coverage,
coverage_curve (list), operator_counts (list of 8).

Aggregate outputs:
1. **Success rate** per agent per case: fraction of 5 runs with a verified crash.
2. **Executions to first verified crash:** median over successful runs.
3. **Final coverage:** mean ± std.
4. **Coverage AUC:** area under the coverage-vs-executions curve.
5. **Operator usage:** share of each operator (answers RQ3).

Plots: coverage-vs-executions line per agent (mean, with a shaded band for
spread); a bar chart of operator usage.

Statistics: Q-learning vs each baseline using the two-sided
Mann–Whitney U test (α = 0.05) on final coverage and executions-to-crash,
plus the Vargha–Delaney A12 effect size. Use `scipy.stats.mannwhitneyu`.

---

## 10. Build order (do these in sequence)

1. **Feasibility (case 12950):** pull both images; confirm `-vul` crashes
   and `-fix` does not; find the fuzz binary; confirm coverage prints.
   (Script `feasibility_gate.sh` already exists.)
2. `mutators.py` + a unit test that each operator returns valid, non-empty bytes.
3. `ArvoEnv` with `reset`/`step`/`_run_input`; test the random agent end to end on 12950.
4. `_verify` against the `-fix` image.
5. The three agents.
6. `run_experiment.py` for one case, all three agents, Setting B.
7. Scale to all six cases and both settings.
8. `metrics.py`: tables, plots, stats.

Stop at step 1 and report back if the feasibility gate fails. Everything
after it depends on the container behaving as expected.

---

## 11. Research questions the results must answer
- **RQ1:** Does Q-learning reach higher coverage (AUC) than Random?
- **RQ2:** Does Q-learning reach a verified crash in fewer executions than
  Random and UCB1 (Setting B)?
- **RQ3:** Which operators does the learned policy prefer per input format?

---

## 12. Honesty rules (non-negotiable for this assignment)
- Never fabricate a result, a coverage number, or a crash.
- If something cannot run, record it and use the stated fallback.
- Report negative results (e.g., Q-learning not beating Random) plainly.
- Keep random seeds fixed (0–4) so every run is reproducible.
