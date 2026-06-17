# Team Bundle — scm_bench Submission

This directory is your team's submission. It contains exactly four agents
(`retailer`, `wholesaler`, `distributor`, `factory`) and one `manifest.json`
that names them. The operator's evaluator runs the same four-agent
contract against every team's bundle, on the same scenarios, with the
same seeds. Anything outside this contract is invisible to the evaluator.

## What an agent IS in this framework

An agent is a **governed runtime entity**. Each tick the harness:

1. Builds a `LocalObservation` from the engine state — what your tier
   can legitimately see (its own inventory, backlog, the most recent
   order it received, the shipment about to arrive, the pipeline of
   shipments behind it, a bounded window of order/shipment history,
   and its costs to date). It does **not** include other tiers' state
   or the demand process.
2. Delivers your tier's `inbox` — typed `Message`s that other tiers
   sent you (Phase 2; in Phase 1 the inbox is always empty).
3. Calls your `Agent.step(observation, inbox, t)` and expects an
   `AgentDecision` back.
4. Pulls `decision.order_qty` into the engine.
5. (Phase 2) Routes any `decision.messages` through the typed message
   bus subject to the scenario's communication policy.

What you control is **only** the body of `step()` (and any local memory
or helper methods your `Agent` holds). What you do not control: the
demand function, other tiers' decisions, the engine's lead-time queues,
or any global state.

## What is forbidden

- Reaching outside `LocalObservation` (no global imports, no reading
  files in the bundle to fake a side channel, no peeking at the
  scenario, no invoking the demand function yourself).
- Communicating with other tiers outside of `decision.messages` —
  no shared module-level variables, no writing to disk, no network.
- Declaring tools or message types in `agent.yaml` that you do not
  actually use, or using ones you did not declare. The validator
  checks both directions.
- Exceeding the memory quota declared in `agent.yaml`.

The validator rejects bundles that violate the schema before they ever
run. The runtime rejects decisions that violate the contract during a
run.

## Files

```
team_bundle/
├── manifest.json                  # team_id, sdk_version, agent dirs
├── retailer/{agent.py,agent.yaml}
├── wholesaler/{agent.py,agent.yaml}
├── distributor/{agent.py,agent.yaml}
├── factory/{agent.py,agent.yaml}
└── tests/test_local.py            # runs `scm-bench test-bundle` for you
```

## Workflow

1. Edit the `step()` body in each `<role>/agent.py`. Start with one
   tier, get it working, then move on.
2. If you add memory, set `memory_mode: bounded_buffer` (or `episodic`)
   in `<role>/agent.yaml` and pick a `memory_max_entries`.
3. If you emit messages from `step()`, declare the types in
   `supports_messages`.
4. Run `scm-bench test-bundle .` from this directory. It validates the
   manifests, imports the entrypoints, and runs a 5-tick smoke
   simulation on `intro_step_demand`.
5. Zip this directory flat (no extra parent folder) and submit to the
   operator.

## Beyond Mirror

The four starter `step()` bodies all return
`AgentDecision(order_qty=observation.incoming_order_qty)` — the
canonical "Mirror" policy. It is correct and contract-compliant, and
it produces the textbook bullwhip effect. Replace it. Some directions
to explore:

- **Base stock**: order to bring `inventory_on_hand + incoming_shipment_qty + pipeline_inventory - backlog` up to a target.
- **Moving average**: smooth your incoming-order signal across a
  bounded window (you'll need `memory_mode: bounded_buffer`).
- **Communicating forecast**: emit a `forecast` `Message` upstream so
  the next tier can pre-compensate (Phase 2).

The reference implementations of each live in
`scm_bench.starters.*` — read them, do not import them into
your submission.
