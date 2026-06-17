| Task | Skill | Use when |
| --- | --- | --- |
| Install the bench and verify the checkout | `install` | Preparing a local env before running scenarios or validating a bundle. |
| Drive the bench and read the verdict | `run` | Validating a team bundle, running a scenario or the full matrix, or starting a new team. |

These skills operate the bench. They do not change the engine, scenarios,
metrics, or validator; the code decides the result from declared inputs and the
reported `composite` verdict (lower is better).
