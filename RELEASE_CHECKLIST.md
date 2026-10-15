# Release checklist

## Confirmed by local checks

- [x] Full manuscript PDF and source withheld from current checkout pending arXiv preprint.
- [x] Previously detected mismatch between manuscript and TeX is not presented as a reproducible source release.
- [x] Original numerical scripts and frozen reference data retained.
- [x] Complete numerical experiment re-run and seven figures regenerated.
- [x] Core API, three executable demos, 41 tests, and docs link validation.
- [x] Bilingual READMEs, figure map, result/assumption docs, software citation and MIT license.
- [x] Local wheel build validated.
- [x] Local P0 and P1 commits separated in Git history.

## Required for a *public* release

- [ ] Create an empty **public** repository named `AtheavenMa/The-Horizon-of-Reward-Guidance` under the intended GitHub account. Do not initialize with a README if pushing the supplied history.
- [ ] Push `main` with the two ordered local commits P0 → P1.
- [ ] Check the Actions run on GitHub in Python 3.10, 3.12 and 3.13, plus paper reproduction; resolve any hosted-environment failures.
- [ ] Verify browser README image links, Chinese navigation, and the arXiv coming-soon label.
- [ ] Review GitHub repository description and topics: `reward-modeling`, `reinforcement-learning`, `preference-learning`, `policy-gradient`, `certification`.
- [ ] Double-check the complete manuscript author list and public citation with the authoritative authors before replacing anonymous manuscript metadata.
- [ ] Create a `v0.1.0` tag/release **only after** CI and public links work; do not backdate or claim acceptance at ICLR.

## Publication blockers in the authoring environment

At packaging time, the intended repository did **not** exist and the available GitHub integration exposed repository-file and commit operations but **not new repository creation**. The local container also did not have GitHub DNS/network access or an authenticated `gh` CLI. Thus a local ZIP or Git bundle cannot honestly be presented as a pushed GitHub repository. Publish using a verified GitHub session or create the empty destination first.
