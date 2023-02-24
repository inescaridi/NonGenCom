import pandas as pd

from prometheus_client import CollectorRegistry
from rich.console import Console
from rich.table import Table


console = Console()
err_console = Console(stderr=True)


def print_prometheus_report(registry: CollectorRegistry):
    title = "Resolve scoring processing time"
    console.rule(f"[bold red]{title}", align="left")
    table = Table(
        caption="Calculation time for each step of processing",
        caption_justify="left",
    )

    table.add_column("Step name", no_wrap=True)
    table.add_column("Step desc.", no_wrap=True)
    table.add_column("Time (s)", no_wrap=True)

    total = 0.

    for metric in registry.collect():
        if metric.type == "summary":
            for sample in metric.samples:
                if sample.name.endswith("_sum"):
                    total += sample.value

                    table.add_row(
                        metric.name, metric.documentation, f"{round(sample.value, 3):.3f}",
                    )
    
    table.add_row("", "Total", f"{round(total, 3):.3f}", style="bold")

    console.print(table)


# def _float_format(f: float) -> str:
#     return f"{f:.4f}"


def print_series(title: str, s: pd.Series):
    console.rule(f"[bold red]{title}", align="left")
    table = Table()

    if isinstance(s.index, pd.Index):
        names = [s.index.name]
    if isinstance(s.index, pd.MultiIndex):
        names = s.index.names
    else:
        names = []

    for name in names:
        table.add_column(name)
    table.add_column("Value")

    for index, value in s.items():
        row = list(index) + [f"{value}"]
        table.add_row(*row)

    console.print(table)
