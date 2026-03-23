# SAGE-Drone: Graceful Degradation in Autonomous Agents

**Geometric memory-augmented drone navigation without language model dependency.**

Paper: Graceful Degradation in Autonomous Agents: SAGE Memory-Augmented Drone Navigation Without Language Model Dependency (Likov, 2026b)

Companion paper (memory architecture): https://github.com/Ivelin2022/sage
Companion paper DOI: https://doi.org/10.5281/zenodo.19192937
This paper DOI: https://doi.org/10.5281/zenodo.19193273

## What this repository contains

Code for the drone proof-of-concept study. The SAGE architecture is in the companion repository above. This repository contains the drone-specific integration: embedding pipeline, MultiCube domain routing, SAGEDivided working memory, and the LLM fallback system.

## Repository structure

```
sage_memory.py       v1 — MultiCube only, no working memory
sage_memory_v2.py    v2 — SAGEDivided + MultiCube + Consolidation  
ollama_adapter.py    nomic-embed-text encoder + Mistral 7B integration
run_all.py           Runs all 6 test modes (hybrid, SAGE-only, sequence)
sequence_test.py     4-mode sequence encoding evaluation
requirements.txt     Dependencies
```

## Requirements

```
pip install -r requirements.txt
```

Requires Ollama running locally with:
- nomic-embed-text (embedding)
- mistral:7b (language model)

```
ollama pull nomic-embed-text
ollama pull mistral:7b
```

## Running the tests

```bash
# Full test suite (v1 + v2, hybrid + SAGE-only)
python run_all.py

# Sequence encoding evaluation (4 modes)
python sequence_test.py --save-dir ./sage_state_v1
```

## Key results

| Result | Value |
|---|---|
| Default decisions | 0/72 steps |
| SAGE response time | 2.09s avg |
| LLM response time | 7.49s avg |
| Speedup | 3.6x |
| v2 vs v1 meaningful recall | 8/12 vs 5/12 (+60%) |
| Memory footprint | 240MB (v2) |

## Important note on simulation

All experiments use text-command simulation. No physical drone was used. Observations are natural language strings embedded via nomic-embed-text. Sensor integration (CLIP for camera, lidar encoders) is the primary identified next step.

## Citation

```bibtex
@article{likov2026sagedrone,
  title  = {Graceful Degradation in Autonomous Agents: SAGE Memory-Augmented Drone Navigation},
  author = {Likov, Ivelin},
  year   = {2026},
  url    = {https://github.com/Ivelin2022/sage-drone}
}
```

Acknowledgement: Claude (Anthropic) was used as an AI research assistant.
All research decisions and claims are the sole responsibility of the author.
