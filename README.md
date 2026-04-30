# multi-agent-scm-bench

A 4-tier agentic supply chain testbed (retailer → wholesaler → distributor →
factory, i.e. a classical beer game setup). Lower cost wins.

## Why

In SCM, ten strands shift across three eras:

| Strand | Era I: Networks & Relationships | Era II: Platforms & Ecosystems | Era III: Protocols & Agents |
|---|---|---|---|
| Structure | Networks | Platforms / Ecosystems | Protocols |
| Execution | Machine | Automation | Autonomy |
| Power | Control | Co-creation | Agency |
| Methods | Production | Orchestration | Co-design |
| Technology | Force | Information | Intelligence |
| Transformations | Processes | Flows | Evolution |
| Dominant logic | Goods (GDL) | Service (SDL) | Agency (ADL) |
| Anchor actors | Replicators | Mediators | Explorers |
| Comparative advantage | Factor endowments | Institutional arrangements | Programmable capacity |
| Value | Value-in-Exchange | Value-in-Use | Survival |

Right now, we lack benchmarking and simulation for these shifts. This bench offers a starting point with protocols, autonomy, and agency. Future releases may focus on agent co-design, the value of information as the fundamental unit of exchange, agency levels, the impact of intelligence (e.g. different models, modalities), and the role of evolution as survival in an intelligence explosion scenario. Further, adversarial elements require attention.


## Layout

- `src/scm_bench/` — engine, validator, runner, metrics, LLM runtime
- `examples/mirror/` — baseline agent with a deterministic policy
  (proportional-to-incoming, composite 1.000)
- `examples/llm/` — LLM-driven agent via transformers / vLLM /
  OpenAI-compatible (including local Ollama)
- `tests/` — pytest suite

Bundle contracts and run/scoring instructions live in each
`examples/<name>/AGENTS.md`.

```bash
pip install -e .
pytest
```

---

Tools and hardware: [`ATTRIBUTION.md`](ATTRIBUTION.md). Dependencies:
[`pyproject.toml`](pyproject.toml). License: 0BSD
([`LICENSE`](LICENSE)).
