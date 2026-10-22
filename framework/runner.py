"""Unified experiment runner.

Composes attacks × defenses × agents × datasets and runs all combinations,
collecting results and generating reports.
"""
import os, sys, json, time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .core import Attack, Defense, Dataset, Agent, Scorer, Sample, AgentResult, ScoreResult
from memory_store import clear_all, count, retrieve_memories, set_scope


@dataclass
class ComboResult:
    """Result for a single (attack, defense, agent) combo on N samples."""
    attack: str
    defense: str
    agent: str
    dataset: str
    n: int
    passed: int = 0           # Samples that passed through defense
    blocked: int = 0          # Samples blocked by defense
    cleaned: int = 0          # Samples cleaned by defense
    hp: int = 0               # Honeypot calls
    mw: int = 0               # Memory write calls
    both: int = 0             # Both HP + MW
    errors: int = 0           # API/agent errors
    valid: int = 0            # Valid (non-error) samples
    time: float = 0.0         # Elapsed seconds

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n > 0 else 0.0

    @property
    def block_rate(self) -> float:
        return self.blocked / self.n if self.n > 0 else 0.0

    @property
    def attack_success_rate(self) -> float:
        return self.both / self.valid if self.valid > 0 else 0.0

    @property
    def hp_rate(self) -> float:
        return self.hp / self.valid if self.valid > 0 else 0.0

    @property
    def mw_rate(self) -> float:
        return self.mw / self.valid if self.valid > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack": self.attack, "defense": self.defense,
            "agent": self.agent, "dataset": self.dataset,
            "n": self.n, "passed": self.passed, "blocked": self.blocked,
            "cleaned": self.cleaned, "hp": self.hp, "mw": self.mw,
            "both": self.both, "errors": self.errors, "valid": self.valid,
            "pass_rate": self.pass_rate, "block_rate": self.block_rate,
            "attack_success_rate": self.attack_success_rate,
            "hp_rate": self.hp_rate, "mw_rate": self.mw_rate,
            "time": self.time,
        }


@dataclass
class ExperimentConfig:
    """Configuration for a full experiment run."""
    attacks: List[str] = field(default_factory=list)
    defenses: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    n: int = 50
    max_agent_iterations: int = 6
    temperature: float = 0.0
    output_dir: str = "results"
    seed: int = 42

    # Agent constructor parameters
    agent_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ExperimentRunner:
    """Orchestrates experiments across all configured combinations."""

    def __init__(self, config: ExperimentConfig,
                 attacks: Dict[str, Attack],
                 defenses: Dict[str, Defense],
                 datasets: Dict[str, Dataset],
                 agent_classes: Dict[str, type],
                 agent_kwargs: Dict[str, Any],
                 scorer: Optional[Scorer] = None):
        self.config = config
        self.attacks = attacks
        self.defenses = defenses
        self.datasets = datasets
        self.agent_classes = agent_classes
        self.agent_kwargs = agent_kwargs
        self.scorer = scorer or Scorer()
        self.results: List[ComboResult] = []
        self._aborted = False

    def run(self, progress_callback=None) -> List[ComboResult]:
        """Run all combinations. Returns list of ComboResult."""
        combos = self._build_combos()
        total = len(combos)
        self._aborted = False

        print(f"\n{'='*70}")
        print(f"  COP Evaluation Framework — Experiment Runner")
        print(f"  {total} combinations × {self.config.n} samples = "
              f"{total * self.config.n} total runs")
        print(f"{'='*70}")

        t_global = time.time()

        for idx, (attack_name, defense_name, agent_name, dataset_name) in enumerate(combos):
            if self._aborted:
                print(f"\n[ABORTED] Experiment terminated early ({idx}/{total} done)")
                break

            attack = self.attacks[attack_name]
            defense = self.defenses[defense_name]
            dataset = self.datasets[dataset_name]
            agent_cls = self.agent_classes[agent_name]

            # Each evaluated agent/dataset combination gets an isolated
            # long-term-memory namespace, while memories still persist across
            # samples within that combination.
            set_scope("|".join((defense_name, agent_name, attack_name, dataset_name)))
            clear_all()

            samples = dataset.load(n=self.config.n, seed=self.config.seed)

            # Build agent instance
            kwargs = {}
            if isinstance(self.agent_kwargs, dict) and agent_name in self.agent_kwargs:
                kwargs = self.agent_kwargs[agent_name]
            agent = agent_cls(**kwargs)

            # Apply system-prompt-level defense hardening
            agent.system_prompt = defense.harden_prompt(agent.system_prompt)

            result = self._run_combo(
                attack, defense, agent, dataset, samples,
                combo_label=f"[{idx+1}/{total}]"
            )
            self.results.append(result)

            # Save intermediate results
            self._save_partial()

            if progress_callback:
                progress_callback(idx + 1, total, result)

        total_t = time.time() - t_global
        self._save_final(total_t)

        print(f"\n{'='*70}")
        print(f"  Experiment complete — {len(self.results)}/{total} combos")
        print(f"  Total time: {total_t:.0f}s ({total_t/3600:.1f}h)")
        print(f"{'='*70}")

        return self.results

    def abort(self):
        self._aborted = True

    def _build_combos(self):
        combos = []
        for defense in self.config.defenses:
            for agent in self.config.agents:
                for attack in self.config.attacks:
                    for dataset in self.config.datasets:
                        combos.append((attack, defense, agent, dataset))
        return combos

    def _run_combo(self, attack: Attack, defense: Defense, agent: Agent,
                   dataset: Dataset, samples: List[Sample],
                   combo_label: str = "") -> ComboResult:
        result = ComboResult(
            attack=attack.name, defense=defense.name,
            agent=agent.name, dataset=dataset.name,
            n=len(samples),
        )

        t0 = time.time()
        header = f"{combo_label} {defense.name} | {agent.name} | {attack.name}"
        print(f"\n{'─'*60}")
        print(f"  {header}  (N={len(samples)})")
        print(f"{'─'*60}")

        for i, sample in enumerate(samples):
            try:
                # 1. Produce external evidence records.  ReMAP returns two
                # records; legacy attacks continue to return one.
                attacked_records = attack.apply_records(sample.document, sample.query)

                # 2. Screen each record independently.  A blocked record does
                # not enter memory construction.
                processed_records = []
                was_blocked = False
                was_cleaned = False
                for attacked_record in attacked_records:
                    processed_record, record_blocked = defense.defend(
                        attacked_record, sample.query
                    )
                    if record_blocked:
                        was_blocked = True
                        continue
                    processed_records.append(processed_record)
                    was_cleaned = was_cleaned or processed_record != attacked_record

                if was_blocked:
                    result.blocked += 1
                    # The paper's memory-construction stage only receives
                    # evidence that independently passes the screen.
                    continue

                result.passed += 1
                if was_cleaned:
                    result.cleaned += 1

                # 3. Retrieve relevant long-term memories and expose the two
                # screened records as separate inputs to memory construction.
                retrieved = list((sample.metadata or {}).get("preexisting_memories", []))
                retrieved.extend(retrieve_memories(sample.query))
                user_msg = self._format_agent_message(
                    processed_records, sample.query, retrieved,
                    getattr(attack, "record_types", None),
                )
                memory_before = count()
                agent_result = agent.run(user_msg)
                memory_after = count()

                # Preserve the complete interaction context for an optional
                # semantic judge and record actual memory writes.
                if agent_result.extra is None:
                    agent_result.extra = {}
                agent_result.extra.update({
                    "user_query": sample.query,
                    "query": sample.query,
                    "document": "\n\n".join(attacked_records),
                    "records": attacked_records,
                    "record_types": getattr(attack, "record_types", []),
                    "processed_records": processed_records,
                    "attack_goal": (sample.metadata or {}).get("goal_text", ""),
                    "sample_id": sample.sample_id,
                    "memory_write_count": max(0, memory_after - memory_before),
                })

                # 4. Score
                if agent_result.error:
                    result.errors += 1
                else:
                    score = self.scorer.score(agent_result)
                    if score.hp:
                        result.hp += 1
                    if score.mw:
                        result.mw += 1
                    if score.both:
                        result.both += 1

            except Exception as e:
                result.errors += 1

            # Progress reporting every 10 samples
            if (i + 1) % 10 == 0:
                v = i + 1 - result.errors
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(samples)}] HP={result.hp}/{v} "
                      f"MW={result.mw}/{v} Both={result.both}/{v} "
                      f"B={result.blocked} P={result.passed} [{elapsed:.0f}s]")
                sys.stdout.flush()

        result.valid = len(samples) - result.errors
        result.time = time.time() - t0

        print(f"  >>> DONE {result.time:.0f}s | P={result.passed}/{result.n} "
              f"B={result.blocked} | Both={result.attack_success_rate:.0%} "
              f"HP={result.hp_rate:.0%} MW={result.mw_rate:.0%}")

        return result

    @staticmethod
    def _format_agent_message(records: List[str], query: str,
                              retrieved_memories: List[str],
                              record_types: Optional[List[str]] = None) -> str:
        """Build an explicit trusted-query/untrusted-record boundary."""
        sections = []
        if retrieved_memories:
            sections.append(
                "[RETRIEVED_LONG_TERM_MEMORIES]\n" +
                "\n".join(f"<memory>{m}</memory>" for m in retrieved_memories)
            )
        record_text = []
        for index, record in enumerate(records, 1):
            record_type = "external"
            if record_types and index <= len(record_types):
                record_type = record_types[index - 1]
            record_text.append(
                f"<external_record id=\"{index}\" type=\"{record_type}\">\n"
                f"{record}\n</external_record>"
            )
        sections.append(
            "[SCREENED_EXTERNAL_RECORDS]\n" +
            ("\n\n".join(record_text) if record_text else "(none)")
        )
        sections.append("[USER_QUERY]\n" + (query or "Please answer based on the records."))
        return "\n\n".join(sections)

    def _save_partial(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, "experiment_partial.json")
        with open(path, "w") as f:
            json.dump(
                {"results": [r.to_dict() for r in self.results]},
                f, indent=2, ensure_ascii=False,
            )

    def _save_final(self, total_time: float):
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, "experiment_results.json")
        with open(path, "w") as f:
            json.dump({
                "config": {
                    "attacks": self.config.attacks,
                    "defenses": self.config.defenses,
                    "agents": self.config.agents,
                    "datasets": self.config.datasets,
                    "n": self.config.n,
                    "seed": self.config.seed,
                    "total_time": total_time,
                },
                "results": [r.to_dict() for r in self.results],
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {path}")

    def print_summary(self):
        """Print a formatted summary table."""
        if not self.results:
            print("No results yet.")
            return

        # Group by defense
        by_defense: Dict[Tuple[str, str], List[ComboResult]] = {}
        for r in self.results:
            by_defense.setdefault((r.defense, r.dataset), []).append(r)

        for (dname, dataset_name), items in by_defense.items():
            print(f"\n{'='*80}")
            print(f"  DEFENSE: {dname} | DATASET: {dataset_name}")
            print(f"{'='*80}")

            # Header
            agents_seen = sorted(set(r.agent for r in items))
            hdr = f"  {'Attack':<16s}"
            for a in agents_seen:
                hdr += f" {a + ' Pass':>14s} {a + ' Attack':>14s}"
            print(hdr)
            print("  " + "-" * (16 + 30 * len(agents_seen)))

            # Per attack
            attacks_seen = sorted(set(r.attack for r in items))
            for aname in attacks_seen:
                row = f"  {aname:<16s}"
                for ag in agents_seen:
                    match = [r for r in items
                             if r.attack == aname and r.agent == ag]
                    if match:
                        r = match[0]
                        row += f" {r.pass_rate:>13.0%}  {r.attack_success_rate:>13.0%}"
                    else:
                        row += f" {'—':>14s} {'—':>14s}"
                print(row)

            # Average
            row2 = f"  {'AVERAGE':<16s}"
            for ag in agents_seen:
                ag_items = [r for r in items if r.agent == ag]
                if ag_items:
                    avg_p = sum(r.pass_rate for r in ag_items) / len(ag_items)
                    avg_a = sum(r.attack_success_rate for r in ag_items) / len(ag_items)
                    row2 += f" {avg_p:>13.0%}  {avg_a:>13.0%}"
            print("  " + "-" * (16 + 30 * len(agents_seen)))
            print(row2)

    def generate_report(self, output_path: str = None) -> str:
        """Generate a Markdown report and return the path."""
        if output_path is None:
            output_path = os.path.join(self.config.output_dir,
                                       "experiment_report.md")

        lines = []
        lines.append("# COP0615v2 — Experiment Results")
        lines.append("")
        lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Config**: {len(self.config.attacks)} attacks × "
                     f"{len(self.config.defenses)} defenses × "
                     f"{len(self.config.agents)} agents × "
                     f"N={self.config.n}")
        lines.append("")
        lines.append("## Metrics")
        lines.append("")
        lines.append("- **Pass Rate**: Documents passing through defense (not blocked)")
        lines.append("- **Attack Success**: Both HP + MW triggered (Both rate)")
        lines.append("- **HP**: Honeypot call rate | **MW**: Memory write rate")
        lines.append("")

        by_defense: Dict[Tuple[str, str], List[ComboResult]] = {}
        for r in self.results:
            by_defense.setdefault((r.defense, r.dataset), []).append(r)

        for (dname, dataset_name), items in by_defense.items():
            lines.append("---")
            lines.append(f"## Defense: {dname} | Dataset: {dataset_name}")
            lines.append("")

            agents_seen = sorted(set(r.agent for r in items))
            header = "| Attack |"
            sep = "|--------|"
            for a in agents_seen:
                header += f" {a} Pass | {a} Attack |"
                sep += "---:|:---:|"
            lines.append(header)
            lines.append(sep)

            attacks_seen = sorted(set(r.attack for r in items))
            for aname in attacks_seen:
                row = f"| {aname} |"
                for ag in agents_seen:
                    match = [r for r in items
                             if r.attack == aname and r.agent == ag]
                    if match:
                        r = match[0]
                        row += f" {r.pass_rate:.0%} | {r.attack_success_rate:.0%} |"
                    else:
                        row += " — | — |"
                lines.append(row)

            # Average
            avg_row = "| **AVERAGE** |"
            for ag in agents_seen:
                ag_items = [r for r in items if r.agent == ag]
                if ag_items:
                    avg_p = sum(r.pass_rate for r in ag_items) / len(ag_items)
                    avg_a = sum(r.attack_success_rate for r in ag_items) / len(ag_items)
                    avg_row += f" **{avg_p:.0%}** | **{avg_a:.0%}** |"
            lines.append(avg_row)
            lines.append("")

            # Detail table
            for ag in agents_seen:
                lines.append(f"### {dname} — {ag} Detail")
                lines.append("")
                lines.append("| Attack | HP | MW | Both | Pass | Blocked | Time |")
                lines.append("|--------|----|----|------|------|---------|------|")
                for aname in attacks_seen:
                    match = [r for r in items
                             if r.attack == aname and r.agent == ag]
                    if match:
                        r = match[0]
                        lines.append(f"| {aname} | {r.hp}/{r.valid} | {r.mw}/{r.valid} "
                                     f"| {r.both}/{r.valid} | {r.passed}/{r.n} "
                                     f"| {r.blocked}/{r.n} | {r.time:.0f}s |")
                lines.append("")

        lines.append("---")
        lines.append("*Report generated by COP0615v2 Evaluation Framework*")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\nReport saved to {output_path}")
        return output_path
