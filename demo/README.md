# Explore reward guidance

**No installation:** save [horizon_playground.html](horizon_playground.html) and open the saved file in any browser. Try changing how quickly the policy changes, how fast the reward can adapt, and when more feedback arrives.

The default corresponds to the controlled example: the original feedback supports progress until time **5**; new feedback becomes available at time **4**, and the illustrated trajectory remains supported through time **15**. Moving the sliders explores other mathematical settings. It does not simulate new human-label collections.

For the reproducible Python demos:

~~~bash
python -m pip install -r requirements.txt
python demo/demo.py
python demo/demo_feedback.py
python demo/demo_ordinary_pg.py
~~~

Each program saves a plot and JSON record in `outputs/`.

[Main findings](../docs/RESULTS.md) · [Theory](../docs/THEORY.md)
