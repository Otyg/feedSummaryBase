# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse, promote, or sell copies of
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from feedsummary_core.persistence import MongoDBStore, create_store
from feedsummary_core.tagging_ml.benchmark import BenchmarkSettings, run_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark classical multilabel models against historical MongoDB tags."
    )
    parser.add_argument("--config", required=True, help="FeedSummary YAML configuration")
    parser.add_argument(
        "--output-dir", default="artifacts/tag-benchmark", help="Parent directory for run outputs"
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Allowed tag category; repeat for multiple categories (default: GENERAL)",
    )
    parser.add_argument(
        "--min-label-support", type=int, help="Minimum positive articles per label (default: 10)"
    )
    parser.add_argument("--max-tags", type=int, help="Maximum predicted tags per article")
    parser.add_argument("--max-text-chars", type=int, help="Maximum input characters per article")
    parser.add_argument("--max-articles", type=int, help="Optional export limit")
    parser.add_argument("--seed", type=int, help="Random split and estimator seed")
    parser.add_argument("--n-jobs", type=int, help="Parallel one-vs-rest workers")
    parser.add_argument(
        "--max-category-combinations",
        type=int,
        help="Safety limit for non-empty category subsets (default: 63)",
    )
    return parser


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    return config


def _settings(config: dict[str, Any], args: argparse.Namespace) -> BenchmarkSettings:
    tagging = config.get("tagging") if isinstance(config.get("tagging"), dict) else {}
    ml_config = tagging.get("ml") if isinstance(tagging.get("ml"), dict) else {}
    categories = args.categories or ml_config.get("categories") or ["GENERAL"]
    if isinstance(categories, str):
        categories = [categories]
    configured_max_articles = (
        args.max_articles if args.max_articles is not None else ml_config.get("max_articles")
    )
    return BenchmarkSettings(
        categories=tuple(categories),
        min_label_support=(
            args.min_label_support
            if args.min_label_support is not None
            else int(ml_config.get("min_label_support", 10))
        ),
        max_tags_per_article=(
            args.max_tags
            if args.max_tags is not None
            else int(ml_config.get("max_tags_per_article", 5))
        ),
        max_text_chars=(
            args.max_text_chars
            if args.max_text_chars is not None
            else int(ml_config.get("max_text_chars", 20_000))
        ),
        max_articles=(
            int(configured_max_articles) if configured_max_articles is not None else None
        ),
        random_seed=(args.seed if args.seed is not None else int(ml_config.get("random_seed", 42))),
        n_jobs=(args.n_jobs if args.n_jobs is not None else int(ml_config.get("n_jobs", 1))),
        max_category_combinations=(
            args.max_category_combinations
            if args.max_category_combinations is not None
            else int(ml_config.get("max_category_combinations", 63))
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    store: MongoDBStore | None = None
    raw_uri = ""
    try:
        config = _load_config(args.config)
        store_config = config.get("store")
        if not isinstance(store_config, dict):
            raise TypeError("Configuration must contain a store mapping")
        provider = str(store_config.get("provider") or store_config.get("type") or "").lower()
        if provider not in {"mongo", "mongodb"}:
            raise ValueError("The first ML benchmark version supports MongoDB only")

        read_only_config = dict(store_config)
        read_only_config["initialize_schema"] = False
        raw_uri = str(read_only_config.get("uri") or "")
        created_store = create_store(read_only_config)
        if not isinstance(created_store, MongoDBStore):  # pragma: no cover - guarded by provider
            raise TypeError("Configured store is not MongoDB")
        store = created_store
        settings = _settings(config, args)
        rows = store.iter_articles_with_tags(
            categories=list(settings.categories), limit=settings.max_articles
        )
        report = run_benchmark(rows, settings, args.output_dir)
        print(f"Categories: {', '.join(report['winner']['categories'])}")
        print(f"Winner: {report['winner']['candidate_id']}")
        print(f"Output: {report['output_directory']}")
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary converts failures to exit codes
        message = str(error)
        if raw_uri:
            message = message.replace(raw_uri, "<mongodb-uri>")
        print(f"Benchmark failed ({type(error).__name__}): {message}", file=sys.stderr)
        return 2
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
