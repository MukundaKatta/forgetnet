"""CLI for ForgetNet AI memory and forgetting research."""

from __future__ import annotations

import numpy as np
import click
from rich.console import Console

from forgetnet.memory.forgetting import ForgettingCurveAnalyzer
from forgetnet.memory.interference import InterferenceDetector
from forgetnet.memory.retention import RetentionTester
from forgetnet.models import ForgetNetReport
from forgetnet.report import print_report
from forgetnet.unlearning.eraser import KnowledgeEraser

console = Console()


@click.group()
def cli() -> None:
    """ForgetNet - AI Memory and Forgetting Research."""
    pass


@cli.command()
@click.option("--num-facts", default=10, help="Number of facts to test")
@click.option("--max-distance", default=5000, help="Max context distance")
def retention(num_facts: int, max_distance: int) -> None:
    """Run retention decay simulation."""
    console.print("[bold]Simulating retention decay...[/]")
    tester = RetentionTester()
    points = tester.simulate_retention_decay(
        num_facts=num_facts, max_distance=max_distance
    )
    analyzer = ForgettingCurveAnalyzer()
    curve = analyzer.fit_ebbinghaus(points)
    half = analyzer.half_life(curve)
    report = ForgetNetReport(forgetting_curve=curve)
    print_report(report, console)
    console.print(f"\nMemory half-life: [bold green]{half:.1f}[/] context tokens")


@cli.command()
@click.option("--similarity", default=0.7, help="Similarity between items")
def interference(similarity: float) -> None:
    """Run interference simulation."""
    console.print("[bold]Simulating memory interference...[/]")
    detector = InterferenceDetector()
    pro = detector.simulate_interference(
        similarity=similarity, interference_type="proactive"
    )
    retro = detector.simulate_interference(
        similarity=similarity, interference_type="retroactive"
    )
    report = ForgetNetReport(interference_results=[pro, retro])
    print_report(report, console)


@cli.command()
@click.option("--num-facts", default=3, help="Number of facts to unlearn")
@click.option("--dim", default=16, help="Embedding dimension")
def unlearn(num_facts: int, dim: int) -> None:
    """Run knowledge unlearning simulation."""
    console.print("[bold]Simulating knowledge unlearning...[/]")
    rng = np.random.RandomState(42)
    weights = rng.randn(dim).astype(np.float32)
    eraser = KnowledgeEraser()
    results = []
    for i in range(num_facts):
        emb = rng.randn(dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # normalize
        result, weights = eraser.erase(
            target_fact=f"fact_{i}",
            fact_embedding=emb,
            weights=weights,
        )
        results.append(result)
    report = ForgetNetReport(unlearning_results=results)
    print_report(report, console)


if __name__ == "__main__":
    cli()
