# Third-party notices and provenance

This repository includes research software, numerical results, and images derived from the uploaded `Horizon(1).zip`. The original ICLR style files, conference templates, and third-party `natbib.sty` / `fancyhdr.sty` files in that ZIP are **not** redistributed in this release. No full manuscript PDF or TeX source is included while the preprint is pending (**arXiv coming soon**). The root MIT license applies to original software code and project documentation; it does **not** purport to relicense the forthcoming manuscript or third-party material.

Runtime dependencies are installed from their own upstream projects, not vendored:

- [NumPy](https://numpy.org/) for numerical arrays.
- [SciPy](https://scipy.org/) for numerical solvers, statistics and linear programming.
- [Matplotlib](https://matplotlib.org/) for numerical plots.
- [pytest](https://pytest.org/) for test execution.

Each project retains its own license. Check installed package metadata for the exact terms applicable to the versions you use.

The code includes standard mathematical techniques such as Bradley–Terry binary comparisons, Hoeffding-type bounds, and linear-program feasibility checks. These mathematical tools are not represented as original inventions by this repository. See the manuscript for citations, proof hypotheses, and related-work discussion.
