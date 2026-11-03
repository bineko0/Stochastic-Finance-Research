"""Save figures and tables under reports/, so every number in the report comes from code."""
from __future__ import annotations

import pandas as pd

from sfr.config import project_path


def save_table(cfg: dict, name: str, df: pd.DataFrame, index: bool = False):
    path = project_path(cfg, "reports") / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"  table  -> {path.relative_to(project_path(cfg, 'reports').parent)}")
    return path


def save_figure(cfg: dict, name: str, fig):
    import matplotlib.pyplot as plt

    path = project_path(cfg, "reports") / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure -> {path.relative_to(project_path(cfg, 'reports').parent)}")
    return path


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])
