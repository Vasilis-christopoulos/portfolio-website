#!/usr/bin/env python3
"""Evaluate repo/profile retrieval quality against a small labeled dataset."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.messages import HumanMessage

from app import DEFAULT_LIMIT, build_agent_graph, retrieve_profile_context, search_showcase_repos


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dedupe_preserve(items: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def calc_metrics(expected: List[str], predicted: List[str], k: int) -> Optional[Dict[str, float]]:
    expected_set = {item for item in expected if item}
    if not expected_set:
        return None
    ranked = dedupe_preserve([item for item in predicted if item])[:k]
    hits = [1 if item in expected_set else 0 for item in ranked]
    recall = sum(hits) / len(expected_set)
    mrr = 0.0
    for idx, item in enumerate(ranked, start=1):
        if item in expected_set:
            mrr = 1.0 / idx
            break
    dcg = 0.0
    for idx, rel in enumerate(hits, start=1):
        if rel:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return {"recall@k": recall, "mrr": mrr, "ndcg@k": ndcg}


def mean_metric(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def maybe_run_ragas(_: List[Dict[str, Any]]) -> None:
    print("RAGAS: skipped (requires manual setup with reference answers/contexts).")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def normalize_expected_plan(raw: Any) -> Optional[Dict[str, bool]]:
    if not isinstance(raw, dict):
        return None
    normalized: Dict[str, bool] = {}
    for key, value in raw.items():
        normalized[key] = coerce_bool(value)
    return normalized


def plan_matches(expected: Dict[str, bool], actual: Dict[str, Any]) -> bool:
    for key, value in expected.items():
        if coerce_bool(actual.get(key)) != value:
            return False
    return True


async def run_plan_checks(
    items: List[Dict[str, Any]],
    request_limit: int,
    print_debug: bool = False,
) -> Tuple[List[bool], List[str]]:
    agent = build_agent_graph()
    results: List[bool] = []
    failed_ids: List[str] = []
    for item in items:
        expected_plan = normalize_expected_plan(item.get("expected_plan"))
        expected_plan_any = item.get("expected_plan_any") or []
        if not expected_plan and not expected_plan_any:
            continue
        query = item.get("query", "")
        if not query:
            continue
        session_id = item.get("session_id") or f"eval-{item.get('id', 'unknown')}"
        config = {"configurable": {"thread_id": session_id}}
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)], "request_limit": request_limit},
            config=config,
        )
        plan: Dict[str, Any] = {}
        state_snapshot = None
        if hasattr(agent, "aget_state"):
            state_snapshot = await agent.aget_state(config)
        elif hasattr(agent, "get_state"):
            state_snapshot = agent.get_state(config)
        if state_snapshot is not None:
            state_values = getattr(state_snapshot, "values", state_snapshot)
            if isinstance(state_values, dict):
                plan = state_values.get("plan") or {}
        if not plan:
            plan = result.get("plan") or {}
        matched = False
        if expected_plan and plan_matches(expected_plan, plan):
            matched = True
        if expected_plan_any:
            for candidate in expected_plan_any:
                normalized = normalize_expected_plan(candidate) or {}
                if normalized and plan_matches(normalized, plan):
                    matched = True
                    break
        results.append(matched)
        if not matched:
            failed_ids.append(item.get("id", "unknown"))
            if print_debug:
                print(f"[plan] {item.get('id', 'unknown')} -> {plan}")
    return results, failed_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repo/profile retrieval.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/rag_eval.jsonl"),
        help="Path to JSONL eval dataset.",
    )
    parser.add_argument("--repo-k", type=int, default=5, help="Top-k repos to score.")
    parser.add_argument(
        "--profile-k", type=int, default=5, help="Top-k profile chunks to score."
    )
    parser.add_argument(
        "--print-items",
        action="store_true",
        help="Print per-item metrics.",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Attempt RAGAS evaluation if configured.",
    )
    parser.add_argument(
        "--check-plan",
        action="store_true",
        help="Run the planner and compare to expected_plan fields.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print planner outputs for mismatched items.",
    )
    parser.add_argument(
        "--request-limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Limit passed to the agent when checking plans.",
    )
    args = parser.parse_args()

    load_dotenv()
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("LangSmith tracing enabled via LANGCHAIN_TRACING_V2.")

    repo_metrics: List[Dict[str, float]] = []
    profile_metrics: List[Dict[str, float]] = []
    repo_empty_results: List[bool] = []
    profile_empty_results: List[bool] = []
    items = list(load_jsonl(args.data))

    for item in items:
        query = item.get("query", "")
        if not query:
            continue
        item_id = item.get("id", "unknown")
        expected_repo_ids = item.get("expected_repo_ids") or []
        expected_profile_sources = item.get("expected_profile_sources") or []
        expect_repo_empty = coerce_bool(item.get("expect_repo_empty"))
        expect_profile_empty = coerce_bool(item.get("expect_profile_empty"))

        if expected_repo_ids or expect_repo_empty:
            repo_hits = search_showcase_repos.invoke(
                {"query": query, "limit": args.repo_k}
            )
            predicted_repo_ids = [hit.get("repo_id") for hit in repo_hits if hit]
            if expected_repo_ids:
                metrics = calc_metrics(expected_repo_ids, predicted_repo_ids, args.repo_k)
                if metrics:
                    repo_metrics.append(metrics)
                    if args.print_items:
                        print(f"[repo] {item_id} {metrics}")
            if expect_repo_empty:
                repo_empty_results.append(len(predicted_repo_ids) == 0)

        if expected_profile_sources or expect_profile_empty:
            profile_hits = retrieve_profile_context.invoke(
                {"query": query, "limit": args.profile_k}
            )
            predicted_sources = [hit.get("source") for hit in profile_hits if hit]
            if expected_profile_sources:
                metrics = calc_metrics(
                    expected_profile_sources, predicted_sources, args.profile_k
                )
                if metrics:
                    profile_metrics.append(metrics)
                    if args.print_items:
                        print(f"[profile] {item_id} {metrics}")
            if expect_profile_empty:
                profile_empty_results.append(len(predicted_sources) == 0)

    print("\nSummary:")
    if repo_metrics:
        print(
            "Repos - recall@k: {:.3f}, mrr: {:.3f}, ndcg@k: {:.3f}".format(
                mean_metric([m["recall@k"] for m in repo_metrics]),
                mean_metric([m["mrr"] for m in repo_metrics]),
                mean_metric([m["ndcg@k"] for m in repo_metrics]),
            )
        )
    else:
        print("Repos - no scored items (missing expected_repo_ids).")

    if profile_metrics:
        print(
            "Profile - recall@k: {:.3f}, mrr: {:.3f}, ndcg@k: {:.3f}".format(
                mean_metric([m["recall@k"] for m in profile_metrics]),
                mean_metric([m["mrr"] for m in profile_metrics]),
                mean_metric([m["ndcg@k"] for m in profile_metrics]),
            )
        )
    else:
        print("Profile - no scored items (missing expected_profile_sources).")

    if repo_empty_results:
        print(
            "Repos - empty@k accuracy: {:.3f}".format(
                mean_metric([1.0 if hit else 0.0 for hit in repo_empty_results])
            )
        )
    if profile_empty_results:
        print(
            "Profile - empty@k accuracy: {:.3f}".format(
                mean_metric([1.0 if hit else 0.0 for hit in profile_empty_results])
            )
        )

    if args.check_plan:
        plan_results, failed = asyncio.run(
            run_plan_checks(items, args.request_limit, print_debug=args.print_plan)
        )
        if plan_results:
            print(
                "Plan - accuracy: {:.3f}".format(
                    mean_metric([1.0 if hit else 0.0 for hit in plan_results])
                )
            )
        if failed:
            print("Plan - mismatches: " + ", ".join(failed))

    if args.ragas:
        maybe_run_ragas(items)


if __name__ == "__main__":
    main()
