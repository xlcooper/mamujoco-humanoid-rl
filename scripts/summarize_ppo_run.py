from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a lightweight PPO run summary.")
    parser.add_argument("--run-dir", required=True, help="AutoDL run directory.")
    parser.add_argument("--output", required=True, help="Experiment record markdown path.")
    parser.add_argument("--eval-output", default=None, help="Optional text file from evaluate.py.")
    parser.add_argument("--tail", type=int, default=20, help="Number of metric rows to keep.")
    return parser


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_metrics_tail(path: Path, tail: int) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return reader.fieldnames or [], []

    fieldnames = reader.fieldnames or list(rows[0].keys())
    return fieldnames, rows[-tail:]


def format_table(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No metric rows found.\n"

    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body_lines = []

    for row in rows:
        values = [row.get(field, "") for field in fieldnames]
        body_lines.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *body_lines]) + "\n"


def title_from_output(output_path: Path) -> str:
    words = output_path.stem.replace("-", "_").split("_")
    return " ".join(word.upper() if word == "ppo" else word.capitalize() for word in words)


def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"

    config = read_json(config_path)
    metric_fields, metric_rows = read_metrics_tail(metrics_path, args.tail)

    eval_text = "Evaluation output was not provided.\n"
    if args.eval_output is not None:
        eval_path = Path(args.eval_output)
        eval_text = eval_path.read_text(encoding="utf-8-sig")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_name = output_path.stem
    record_title = title_from_output(output_path)

    content = f"""# {record_title}

## 目的

记录 `{record_name}` 的轻量训练结果，用于本地分析当前手写 PPO baseline 的学习趋势和稳定性。

本记录只提交摘要，不提交 checkpoint、完整 run 目录、TensorBoard events 或视频。

## Run 信息

- run directory: `{run_dir}`
- environment baseline: `AUTODL_HOST_BASELINE.md`

## Config

```json
{json.dumps(config, indent=2, ensure_ascii=False)}
```

## Metrics Tail

保留 `metrics.csv` 最后 {args.tail} 行：

{format_table(metric_fields, metric_rows)}

## Evaluation Output

```text
{eval_text.strip()}
```

## 初步观察

- TODO: 本地 pull 后分析 episodic return、episode length、value loss、entropy、approx KL 和 clip fraction。
- TODO: 判断下一步是否需要 observation normalization / reward scaling。
"""

    output_path.write_text(content, encoding="utf-8")
    print(f"wrote_summary={output_path}")


if __name__ == "__main__":
    main()
