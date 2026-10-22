"""COP Evaluation Framework — unified attack/defense evaluation for LLM Agents.

Architecture:
  datasets/  — Dataset loaders
  attacks/   — Attack methods (including relation-deferred ReMAP records)
  defenses/  — Per-record input and system-prompt defenses
  agents/    — Tool-using agents with long-term memory

All components are auto-discovered via naming convention.
Run `python run_experiment.py` to execute all combinations.
"""
from .core import Attack, Defense, Dataset, Agent, Scorer, Sample, AgentResult, ScoreResult
from .registry import get_attacks, get_defenses, get_datasets, get_agents
from .runner import ExperimentRunner
