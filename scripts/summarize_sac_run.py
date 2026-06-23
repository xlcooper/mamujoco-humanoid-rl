from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a lightweight SB3 SAC run summary.")
    parser.add_argument("--run-dir", required=True, help="AutoDL run directory.")
    parser.add_argument("--output", required=True, help="Experiment record markdown path.")
    parser.add_argument("--eval-output", default=None, help="Optional text file from evaluation.")
    parser.add_argument("--eval-json", default=None, help="Optional JSON file from evaluation.")
    parser.add_argument("--tail", type=int, default=20, help="Number of monitor rows to keep.")
    return parser


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def find_monitor_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*.monitor.csv"))
    if candidates:
        return candidates[0]

    monitor_path = run_dir / "monitor.monitor.csv"
    if monitor_path.exists():
        return monitor_path

    raise FileNotFoundError(f"No SB3 monitor CSV found in {run_dir}.")


def read_monitor_tail(path: Path, tail: int) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    # SB3 Monitor 第一行通常是 JSON header comment，需要跳过。
    csv_lines = [line for line in lines if not line.startswith("#")]
    reader = csv.DictReader(csv_lines)
    rows = list(reader)

    if not rows:
        return reader.fieldnames or [], []

    fieldnames = reader.fieldnames or list(rows[0].keys())
    return fieldnames, rows[-tail:]


def format_table(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No monitor rows found.\n"

    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body_lines = []

    for row in rows:
        values = [row.get(field, "") for field in fieldnames]
        body_lines.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *body_lines]) + "\n"


def title_from_output(output_path: Path) -> str:
    words = output_path.stem.replace("-", "_").split("_")
    return " ".join(word.upper() if word in {"sac", "sb3"} else word.capitalize() for word in words)


def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output)
    config_path = run_dir / "config.json"
    monitor_path = find_monitor_csv(run_dir)

    config = read_json(config_path)
    monitor_fields, monitor_rows = read_monitor_tail(monitor_path, args.tail)

    eval_text = "Evaluation output was not provided.\n"
    if args.eval_output is not None:
        eval_text = Path(args.eval_output).read_text(encoding="utf-8-sig")

    eval_json_text = "Evaluation JSON was not provided."
    if args.eval_json is not None:
        eval_json = read_json(Path(args.eval_json))
        eval_json_text = json.dumps(eval_json, indent=2, ensure_ascii=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_name = output_path.stem
    record_title = title_from_output(output_path)

    content = f"""# {record_title}

## 目的

记录 `{record_name}` 的轻量训练结果，用于确认 SB3 SAC smoke test 链路是否跑通。

本记录只提交摘要，不提交 checkpoint、完整 run 目录、replay buffer、TensorBoard events 或视频。

## Run 信息

- run directory: `{run_dir}`
- environment baseline: `AUTODL_HOST_BASELINE.md`
- monitor csv: `{monitor_path}`

## Config

```json
{json.dumps(config, indent=2, ensure_ascii=False)}
```

## Monitor Tail

保留 SB3 Monitor 最后 {args.tail} 行：

{format_table(monitor_fields, monitor_rows)}

## Evaluation Output

```text
{eval_text.strip()}
```

## Evaluation JSON

```json
{eval_json_text}
```

## 初步观察

- TODO: 本地 pull 后确认训练、checkpoint、VecNormalize、TensorBoard 和 evaluation 链路是否完整。
- TODO: 判断 smoke test 是否可以固化为已完成，并决定下一步进入 SAC 长训还是先修参数/环境问题。
"""

    output_path.write_text(content, encoding="utf-8")
    print(f"wrote_summary={output_path}")


if __name__ == "__main__":
    main()
