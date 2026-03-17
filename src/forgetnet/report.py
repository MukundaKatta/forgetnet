"""Report generation for ForgetNet analysis."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from forgetnet.models import ForgetNetReport


def print_report(report: ForgetNetReport, console: Console | None = None) -> None:
    """Print a rich-formatted ForgetNet report."""
    if console is None:
        console = Console()

    # Retention results
    if report.retention_results:
        console.rule("[bold blue]Retention Test Results")
        table = Table(title="Retention Scores")
        table.add_column("Probe", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Recalled", justify="center")
        table.add_column("Type")
        table.add_column("Distance", justify="right")
        for r in report.retention_results:
            table.add_row(
                r.probe.fact[:40],
                f"{r.retention_score:.3f}",
                "[green]Yes[/]" if r.recalled else "[red]No[/]",
                r.retention_type.value,
                str(r.context_distance),
            )
        console.print(table)

    # Forgetting curve
    if report.forgetting_curve:
        fc = report.forgetting_curve
        console.rule("[bold blue]Forgetting Curve")
        console.print(f"Model: [cyan]{fc.model_type}[/]")
        console.print(f"Stability (S): [yellow]{fc.stability:.4f}[/]")
        console.print(f"Initial strength (B): [yellow]{fc.initial_strength:.4f}[/]")
        console.print(f"R-squared: [green]{fc.r_squared:.4f}[/]")

    # Interference results
    if report.interference_results:
        console.rule("[bold blue]Interference Results")
        for ir in report.interference_results:
            console.print(
                f"[cyan]{ir.interference_type}[/]: "
                f"baseline={ir.baseline_retention:.3f} -> "
                f"post={ir.post_interference_retention:.3f} "
                f"(magnitude={ir.interference_magnitude:.3f})"
            )

    # Unlearning results
    if report.unlearning_results:
        console.rule("[bold blue]Unlearning Results")
        table = Table(title="Unlearning Outcomes")
        table.add_column("Fact", style="cyan")
        table.add_column("Pre-Conf", justify="right")
        table.add_column("Post-Conf", justify="right")
        table.add_column("Steps", justify="right")
        table.add_column("Success", justify="center")
        table.add_column("Collateral", justify="right")
        for ur in report.unlearning_results:
            table.add_row(
                ur.target_fact[:30],
                f"{ur.pre_unlearn_confidence:.3f}",
                f"{ur.post_unlearn_confidence:.3f}",
                str(ur.num_gradient_steps),
                "[green]Yes[/]" if ur.success else "[red]No[/]",
                f"{ur.collateral_damage:.3f}",
            )
        console.print(table)
