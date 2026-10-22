#!/usr/bin/env python3
"""COP0615v2 — Unified Attack-Defense Evaluation Framework.

Usage:
    python run_experiment.py                     # Run all combos (default N=50)
    python run_experiment.py --n 10              # Quick test with N=10
    python run_experiment.py --attacks Sleeper,COP_Improved  # Specific attacks
    python run_experiment.py --defenses "Filter v2"           # Specific defense
    python run_experiment.py --agents "Native ReAct"          # Specific agent
    python run_experiment.py --list                           # List all available components

Architecture:
    datasets/   — Document datasets
    attacks/    — Attack methods (ReMAP emits separate evidence records)
    defenses/   — Per-record input and prompt defenses
    agents/     — Tool-using victim agents with long-term memory
    framework/  — Core framework (registry, runner, memory-aware execution)

All components are auto-discovered. Add new modules by placing files in the
appropriate directory with a class inheriting from the base class.
"""
import os, sys, json, argparse

# Ensure the project root is on the path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

from config import load as load_api_config
from framework.registry import get_attacks, get_defenses, get_datasets, get_agents
from framework.runner import ExperimentRunner, ExperimentConfig


def list_components():
    """List all discovered components."""
    attacks = get_attacks()
    defenses = get_defenses()
    datasets = get_datasets()
    agents = get_agents()

    print("\n" + "=" * 60)
    print("  Available Components")
    print("=" * 60)

    print(f"\n  ATTACKS ({len(attacks)}):")
    for name, obj in attacks.items():
        print(f"    {name:<20s} — {obj.description[:70]}...")

    print(f"\n  DEFENSES ({len(defenses)}):")
    for name, obj in defenses.items():
        print(f"    {name:<20s} — {obj.description[:70]}...")

    print(f"\n  DATASETS ({len(datasets)}):")
    for name, obj in datasets.items():
        print(f"    {name:<20s} — {obj.description[:70]}...")

    print(f"\n  AGENTS ({len(agents)}):")
    for name, obj in agents.items():
        print(f"    {name:<20s} — {obj.description[:70]}...")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="COP0615v2 Unified Attack-Defense Evaluation Framework")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of samples per combo (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Dataset sampling seed (default: 42)")
    parser.add_argument("--max-agent-iterations", type=int, default=6,
                        help="Maximum tool-loop iterations (default: 6)")
    parser.add_argument("--attacks", type=str, default="",
                        help="Comma-separated attack names (default: all)")
    parser.add_argument("--defenses", type=str, default="",
                        help="Comma-separated defense names (default: all)")
    parser.add_argument("--agents", type=str, default="",
                        help="Comma-separated agent names (default: all)")
    parser.add_argument("--datasets", type=str, default="",
                        help="Comma-separated dataset names (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List all available components and exit")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--semantic-judge", action="store_true",
                        help="Use the optional GPT-4o semantic judge for scoring")
    args = parser.parse_args()

    # Discover components
    attacks = get_attacks()
    defenses = get_defenses()
    datasets = get_datasets()
    agent_classes = get_agents()

    if args.list:
        list_components()
        return

    # Validate requested components
    attack_names = ([a.strip() for a in args.attacks.split(",") if a.strip()]
                    if args.attacks else list(attacks.keys()))
    defense_names = ([d.strip() for d in args.defenses.split(",") if d.strip()]
                     if args.defenses else list(defenses.keys()))
    agent_names = ([a.strip() for a in args.agents.split(",") if a.strip()]
                   if args.agents else list(agent_classes.keys()))
    dataset_names = ([d.strip() for d in args.datasets.split(",") if d.strip()]
                     if args.datasets else list(datasets.keys()))

    # Check existence
    for name in attack_names:
        if name not in attacks:
            print(f"Error: Attack '{name}' not found. Use --list to see available.")
            return
    for name in defense_names:
        if name not in defenses:
            print(f"Error: Defense '{name}' not found. Use --list to see available.")
            return
    for name in agent_names:
        if name not in agent_classes:
            print(f"Error: Agent '{name}' not found. Use --list to see available.")
            return
    for name in dataset_names:
        if name not in datasets:
            print(f"Error: Dataset '{name}' not found. Use --list to see available.")
            return

    # Load API config
    api_cfg = load_api_config()
    print(f"\n  API: {api_cfg['model']} @ {api_cfg['base_url']}")

    # Build agent kwargs (shared API config)
    agent_kwargs = {}
    for aname in agent_names:
        agent_kwargs[aname] = {
            "model": api_cfg["model"],
            "api_url": api_cfg["base_url"],
            "api_key": api_cfg["api_key"],
            "max_iterations": args.max_agent_iterations,
        }

    # Build experiment config
    config = ExperimentConfig(
        attacks=attack_names,
        defenses=defense_names,
        agents=agent_names,
        datasets=dataset_names,
        n=args.n,
        seed=args.seed,
        max_agent_iterations=args.max_agent_iterations,
        output_dir=args.output,
        agent_kwargs=agent_kwargs,
    )

    scorer = None
    if args.semantic_judge:
        try:
            from tools.gpt4o_judge import GPT4oJudge
            scorer = GPT4oJudge(model=api_cfg["model"], as_scorer=True)
        except Exception as e:
            print(f"Warning: semantic judge unavailable ({e}); using tool-call scorer.")

    # Print config summary
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT CONFIG")
    print(f"{'='*60}")
    print(f"  Attacks:  {', '.join(attack_names)}")
    print(f"  Defenses: {', '.join(defense_names)}")
    print(f"  Agents:   {', '.join(agent_names)}")
    print(f"  Datasets: {', '.join(dataset_names)}")
    print(f"  N:        {args.n}")
    print(f"  Seed:     {args.seed}")
    print(f"  Combos:   {len(defense_names)}×{len(agent_names)}×"
          f"{len(attack_names)}×{len(dataset_names)} = "
          f"{len(defense_names)*len(agent_names)*len(attack_names)*len(dataset_names)}")
    print(f"  Total:    {len(defense_names)*len(agent_names)*len(attack_names)*len(dataset_names)*args.n} runs")
    print(f"{'='*60}")

    # Run
    runner = ExperimentRunner(
        config=config,
        attacks={k: attacks[k] for k in attack_names},
        defenses={k: defenses[k] for k in defense_names},
        datasets={k: datasets[k] for k in dataset_names},
        agent_classes={k: agent_classes[k] for k in agent_names},
        agent_kwargs=agent_kwargs,
        scorer=scorer,
    )

    try:
        results = runner.run()
        runner.print_summary()
        runner.generate_report()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving partial results...")
        runner.abort()
        runner._save_final(0)
        runner.print_summary()
        runner.generate_report()


if __name__ == "__main__":
    main()
