# Contributing

Contributions to the analytical tools, numerical experiments, and demos are welcome.

## Development setup

```bash
python -m pip install -r requirements_dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

For changes affecting the manuscript's numerical evidence:

```bash
python experiments/reproduce.py --all
```

## Contributing

1. Keep reference CSV/JSON files fixed and write new outputs under `outputs/`.
2. Document the assumptions used by new mathematical or statistical results.
3. Add tests for new functionality and run `python -m pytest -q`.
4. Use `python experiments/reproduce.py --all` to check changes to the numerical pipeline.
5. Describe results and their verification in pull requests.
