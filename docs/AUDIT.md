# Try the reward-guidance audit

Use the **horizon-audit** command to check the behavior of an explicitly described finite reward system or compare two sets of recorded training runs.

- **Finite audit:** recompute reward progress from actual outcome probabilities and reward vectors, rather than trusting a number entered in a report.
- **Experiment comparison:** align two methods by training seed and calculate paired differences and an uncertainty interval.
- **Neural results:** record empirical evidence without claiming a proof of a neural support horizon.

## Install

From the repository:

~~~bash
python -m pip install -e .
horizon-audit --help
~~~

The core commands use the standard library and require **no GPU, login or model download**.

## Check a finite reward model

~~~bash
horizon-audit validate --manifest examples/08_guidance_audit/finite_computed.json
horizon-audit audit --manifest examples/08_guidance_audit/finite_computed.json --out outputs/my_audit
~~~

The example supplies outcome probabilities over 16 possibilities, reward values and an interval of plausible preferences. The tool calculates the finite bound from those values.

Change a reward or change the outcome probabilities while keeping their total equal to 1. Running the command again changes the computed result. A deliberately inconsistent stored certificate is rejected or replaced by the independently computed value.

[Finite example](../examples/08_guidance_audit/finite_computed.json) · [Incorrect-certificate example](../examples/08_guidance_audit/intentionally_false_certificate.json)

## Compare training runs

For each method, supply a folder of JSON run records with a `seed` field and one shared numerical metric, such as `fixed_eval`, measured by the same evaluator.

~~~bash
horizon-audit compare \
  --baseline experiments/horizon_final_08/public_smoke/pythia_source \
  --treatment experiments/horizon_final_08/public_smoke/pythia_bounded_arm \
  --metric fixed_eval \
  --pair-by seed \
  --out outputs/my_comparison
~~~

This creates a JSON summary and CSV with **per-seed differences**, their mean and a 95% paired interval when at least two independent seeds are present. It does not count many prompts from one training seed as independent training experiments.

## What the result means

| Data available | Supported result |
|---|---|
| Finite outcome probabilities, rewards and specified compatible preference set | Computed finite progress for that update |
| Valid fixed-size independent binary comparisons under a specified model | Fixed-design statistical statement with its assumptions |
| Real human feedback or neural-model predictions | Held-out empirical metrics, not a formally certified neural horizon |
| Only a previously entered certificate value | Reported, but not independently verified |

[Python API](API.md) · [Original mathematical results](THEORY.md) · [Full reproduction notes](FINAL_REPRODUCE_08.md)
