"""Command line entry point: science-agent {assess,bench,search,limits}."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from . import knowledge
from .config import Config, load_config


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    llm_update = {
        k: v
        for k, v in {"provider": args.provider, "model": args.model, "base_url": args.base_url}.items()
        if v is not None
    }
    if llm_update:
        cfg.llm = cfg.llm.model_validate({**cfg.llm.model_dump(), **llm_update})
    if getattr(args, "domain", None):
        cfg.domain = args.domain
    if args.no_literature:
        cfg.literature.enabled = False
    if args.no_compute:
        cfg.compute.enabled = False
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    return cfg


def _read_idea(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.idea == "-" or (args.idea is None and not sys.stdin.isatty()):
        return sys.stdin.read()
    if not args.idea:
        sys.exit("Provide an idea as an argument, with --file, or on stdin.")
    return args.idea


def _progress(stage: str) -> None:
    print(f"  · {stage}", file=sys.stderr, flush=True)


def cmd_assess(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline
    from .report import save

    cfg = _apply_overrides(load_config(args.config), args)
    idea = _read_idea(args).strip()
    print(f"Assessing with {cfg.llm.provider}:{cfg.llm.model} ...", file=sys.stderr)
    assessment = Pipeline(cfg, progress=_progress).run(idea)
    path = save(assessment, cfg.output_dir)
    v = assessment.verdict
    print(f"\nVerdict: {v.verdict} (confidence {v.confidence:.0%})")
    print(f"Deciding argument: {v.deciding_argument}")
    print(f"Next step: {v.next_step}")
    print(f"\nReport: {path}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import load_benchmark, run_benchmark, save_summary, summarize
    from .pipeline import Pipeline
    from .report import save

    cfg = _apply_overrides(load_config(args.config), args)
    domain, items = load_benchmark(args.benchmark)
    if domain and not args.domain:
        cfg.domain = domain
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i.id in wanted]
    pipeline = Pipeline(cfg)
    run_dir = cfg.output_dir / f"bench-{datetime.now():%Y%m%d-%H%M%S}"

    def assess(idea: str):
        a = pipeline.run(idea)
        save(a, run_dir)
        return a

    def show(o):
        mark = "PASS" if o.correct else "FAIL"
        conf = f"{o.confidence:.0%}" if o.confidence is not None else "-"
        print(f"{mark}  {o.id:<34} expected {'/'.join(o.expected):<26} got {o.got or 'ERROR'} ({conf}) {o.error}")

    print(f"Running {len(items)} benchmark ideas with {cfg.llm.provider}:{cfg.llm.model}\n")
    summary = summarize(run_benchmark(items, assess, on_result=show))
    save_summary(summary, run_dir / "summary.json")
    print(
        f"\nAccuracy {summary['correct']}/{summary['total']} ({summary['accuracy']:.0%}) · "
        f"rejected wrong ideas {summary['rejected_wrong_ideas']}/{summary['wrong_ideas']} · "
        f"falsely rejected real ideas {summary['falsely_rejected_real_ideas']}/{summary['real_ideas']}"
    )
    print(f"Reports and summary: {run_dir}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .tools.literature import LiteratureSearch

    papers = LiteratureSearch(max_results=args.max_results).search(args.queries)
    if args.json:
        print(json.dumps([p.to_dict() for p in papers], indent=2, ensure_ascii=False))
    else:
        for p in papers:
            cites = f", {p.citations} citations" if p.citations is not None else ""
            print(f"- {p.title} ({p.year or 'n.d.'}, {p.source}{cites})\n  {p.url}\n  {p.abstract[:300]}\n")
    return 0


def cmd_limits(args: argparse.Namespace) -> int:
    if args.list:
        print("\n".join(knowledge.available_domains()))
        return 0
    print(knowledge.render_full(knowledge.load_domain(args.domain or "information_theory")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="science-agent", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def llm_options(p):
        p.add_argument("--config", help="YAML config (default: $SCIENCE_AGENT_CONFIG or ./config.yaml)")
        p.add_argument("--provider", choices=["openai_compatible", "anthropic"])
        p.add_argument("--model")
        p.add_argument("--base-url")
        p.add_argument("--domain", help="knowledge base name or path to a YAML file")
        p.add_argument("--no-literature", action="store_true", help="skip arXiv/Semantic Scholar search")
        p.add_argument("--no-compute", action="store_true", help="skip running generated Python")
        p.add_argument("--output-dir")

    p = sub.add_parser("assess", help="assess one idea and write a report")
    p.add_argument("idea", nargs="?", help="the idea as text, or '-' for stdin")
    p.add_argument("--file", help="read the idea from a text file")
    llm_options(p)
    p.set_defaults(func=cmd_assess)

    p = sub.add_parser("bench", help="run the calibration benchmark")
    p.add_argument("benchmark", nargs="?", default="benchmarks/information_theory.yaml")
    p.add_argument("--only", help="comma-separated idea ids")
    llm_options(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("search", help="search arXiv and Semantic Scholar")
    p.add_argument("queries", nargs="+")
    p.add_argument("--max-results", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("limits", help="print a domain knowledge base")
    p.add_argument("domain", nargs="?")
    p.add_argument("--list", action="store_true", help="list available domains")
    p.set_defaults(func=cmd_limits)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
