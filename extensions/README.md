# Extensions: What If the Choke Moves?

The main project predicts rate at the operating point a well is running at. This folder asks a harder question: **how do rate and wellhead pressure change when the choke is moved?** It is kept separate because it is less certain. Volve has few large choke moves to validate against, and pure ML proved unreliable here.

## Contents

```text
extensions/
├── predictors.py         # Common predict(df) wrappers for Perkins, hybrid and Gradient Boosting
├── system/
│   ├── vlp.py            # VLP interface: DataVLP, TableVLP (CSV), CallableVLP
│   ├── ipr.py            # Anchored linear IPR and productivity-index fit
│   └── nodal.py          # Solves choke + tubing + reservoir for WHP and rate
├── response_test.py      # Choke-response test on real choke moves
├── optimize.py           # Choke optimizer (nodal or fixed-WHP mode)
├── plots.py              # Choke-response and rate-vs-choke figures
└── tests/test_nodal.py   # Nodal solver checks against a hand-solved case
```

Outputs go to `outputs/extensions/` and the trained bundle to `models/extensions/whatif_model.pkl`.

## Running

Run the main pipeline's data preparation first, then, from the project root:

```bash
python -m pytest -q extensions/tests
```

```bash
python -m extensions.response_test
```

```bash
python -m extensions.optimize
```

```bash
python -m extensions.plots
```

## What Calls What

| Command | Calls, in order |
|---|---|
| `extensions.response_test` | `find_choke_events()` → `time_blocks()` → per block: `purged_train_mask()`, `fit_predictors()` (→ `predictors.make_predictors()`) → per event: `vlp.DataVLP.fit()`, `ipr.anchored_ipr()` (→ `fit_pi()`), `nodal.choke_rate_function()`, `nodal.solve_operating_point()` → `_summarize()` |
| `extensions.optimize` | `train_final_model()` fits choke models, plus a `DataVLP` and PI for each downhole well → `find_optimal_choke()` → `rate_vs_choke()` → nodal: `solve_operating_point()` at every choke %, or fixed-WHP fallback → match to today's rate → data-coverage check |
| `extensions.plots` | `plot_extensions()` → `plot_choke_response()`, `plot_rate_vs_choke()` |

## Well System (Nodal) Mode

A choke model on its own gives rate for a known WHP. When the choke moves, WHP moves too, because the reservoir and tubing push back. The nodal solver finds the WHP at which three rates agree:

```
q_choke(WHP, choke %)  =  q_IPR( Pwf = VLP(q, WHP) )
```

- **VLP (tubing):** `DataVLP` fits Pwf − WHP per well from the gauge data, using water cut, GLR, WHP and rate. Error on later data is about 116–261 psi against a 2,600–3,200 psi total. You can plug in your own instead, as `TableVLP.from_csv("prosper_export.csv")` (columns `Q_liquid_STB`, `WHP_psig`, `Pwf_psig`, optionally `Water_Cut`, `GLR_scf_stb`) or as `CallableVLP(fn)`.
- **IPR (reservoir):** a straight line through today's measured point, `q = q_now − PI·(Pwf − Pwf_now)`. PI is fitted per well from day-to-day variation within 30-day windows. Volve gives no reservoir pressure, and unconstrained fits put it at impossible values (9,000–28,000 psi) for three wells, so the line is anchored rather than extrapolated. It is most reliable near today's operating point.
- **Matching:** before searching, the optimizer scales the model so it reproduces today's measured rate at today's choke.
- Nodal mode needs a downhole gauge: F-1 C, F-11 H, F-12 H (2008–2010), F-14 H and F-15 D. Elsewhere the optimizer holds WHP fixed and prints a warning.

```python
from extensions.optimize import find_optimal_choke
from extensions.system.vlp import TableVLP

find_optimal_choke(target_q_liquid=20000, state=state, choke_now=45.9, q_now=19455, pwf_now=3910,
                   well='F-14 H', vlp=TableVLP.from_csv("my_vlp.csv"))   # vlp= is optional
```

## Results

**Choke-response test.** 93 real choke moves in the 5 downhole wells. Each pairs a steady day before the move with the first steady day after it. Choke models use blocked time cross-validation (5 blocks, 14-day purge), and the IPR and VLP use only data from before each event. The moves are small: median 3.5 choke points and 5.5% change in rate.

| Model · mode | Direction correct | Rate-change error (% of rate) | WHP direction correct |
|---|---|---|---|
| Rule: rate moves with the choke | 84.9% | - | - |
| No-change baseline | - | 9.8% | - |
| Perkins · fixed WHP | 84.9% | 6.4% | - |
| Hybrid · fixed WHP | 83.9% | **6.1%** | - |
| Gradient Boosting · fixed WHP | 62.4% | 8.0% | - |
| Perkins · nodal | 80.2% | 6.6% | **97.5%** (24 psi error) |
| Hybrid · nodal | 84.0% | 6.9% | 92.6% (37 psi error) |
| Gradient Boosting · nodal | 66.2% | 7.2% | 72.7% (48 psi error) |

Nodal results cover 81 events (77 for Gradient Boosting). The rest had no valid PI or no balance point. The 95% intervals on direction accuracy are about ±9 points.

**Optimizer example (F-14 H, 2016-07-06).** Today the well makes 19,455 STB/d at 45.9% open. With WHP held fixed, opening to 98% looks like it gives about 29,900 STB/d. In nodal mode, WHP falls toward the downstream pressure (about 395 psig) and rate levels off at about 19,900 STB/d. The well is limited by its reservoir and tubing, not by the choke.

These results use the main project's current ML inputs, including the choke pressure drop and flow index.

## Findings

1. **Pure ML gets the choke response wrong most often.** Gradient Boosting predicts the right direction 62% of the time, well below the simple rule "rate moves with the choke" (85%). Where training data is sparse, it predicts a nearly flat rate as the choke closes. Its better accuracy at predicting rate in the main project does not carry over to predicting changes.
2. **Physics-based models (Perkins, hybrid) match the simple rule on direction** and beat the no-change baseline on the size of the change.
3. **Nodal mode adds WHP prediction** (93–98% direction accuracy with physics-based models). On small moves it doesn't improve the rate prediction. Its value shows in large moves, where fixed WHP overestimates by about 50%, but only 5 recorded moves are 10 points or more, so that can't be measured on Volve.

## Limitations

- The IPR is a straight line anchored at today's point, and its implied reservoir pressure is unrealistically high for some wells. Far from today's choke it likely overstates rate at small openings.
- Downstream pressure is held at today's value. Other wells on the same manifold are not modelled.
- Daily averages blur the day a choke moves. The response test skips that day and compares the steady days on either side.
- Validating large choke moves needs step-rate test data, which Volve does not publish.
