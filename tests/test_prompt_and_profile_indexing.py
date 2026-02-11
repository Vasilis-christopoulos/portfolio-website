"""Tests for response prompt guidance and profile markdown indexing behavior."""

from pathlib import Path
import unittest

from app import (
    PLANNER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    is_behavioral_fit_question,
    should_request_context_clarification,
)
from scripts.index_profile_docs import chunk_profile_section, read_profile_sections


class PromptGuidanceTests(unittest.TestCase):
    def test_system_prompt_includes_concise_and_formatting_guidance(self) -> None:
        expected_phrases = (
            "Default to concise responses",
            "Start with a direct answer",
            "short markdown bullet points",
            "single-level bullet points",
            "For motivational or fit questions",
            "Use projects/experience as brief support",
            "Mention specific companies, projects, tools, or metrics only when they appear in the provided context",
        )
        for phrase in expected_phrases:
            self.assertIn(phrase, SYSTEM_PROMPT)

    def test_planner_prompt_covers_why_consulting(self) -> None:
        self.assertIn("why consulting", PLANNER_SYSTEM_PROMPT)

    def test_behavioral_question_detection(self) -> None:
        self.assertTrue(is_behavioral_fit_question("Why consulting?"))
        self.assertTrue(is_behavioral_fit_question("Tell me about yourself."))
        self.assertFalse(is_behavioral_fit_question("Which projects use RAG?"))

    def test_missing_context_guardrail_when_retrieval_empty(self) -> None:
        plan = {"use_profile_search": True, "use_repo_search": False, "should_compare": False}
        self.assertTrue(
            should_request_context_clarification(
                plan=plan,
                repo_search_context=[],
                profile_context=[],
                repo_summaries=[],
            )
        )

    def test_missing_context_guardrail_not_triggered_with_context(self) -> None:
        plan = {"use_profile_search": True, "use_repo_search": False, "should_compare": False}
        self.assertFalse(
            should_request_context_clarification(
                plan=plan,
                repo_search_context=[],
                profile_context=[{"content": "Example context"}],
                repo_summaries=[],
            )
        )

    def test_missing_context_guardrail_not_triggered_with_repo_summaries(self) -> None:
        plan = {"use_profile_search": False, "use_repo_search": True, "should_compare": False}
        self.assertFalse(
            should_request_context_clarification(
                plan=plan,
                repo_search_context=[],
                profile_context=[],
                repo_summaries=[{"repo_id": "x", "title": "Project X"}],
            )
        )


class ProfileIndexingTests(unittest.TestCase):
    def test_master_cv_markdown_splits_into_named_sections(self) -> None:
        cv_path = Path("docs/Vasileios_Christopoulos_Master.md")
        sections = read_profile_sections(cv_path)

        self.assertGreaterEqual(len(sections), 6)
        self.assertTrue(
            any("Section: Professional Experience" in section for section in sections)
        )
        self.assertTrue(any("Section: Education" in section for section in sections))

    def test_master_cv_sections_are_chunkable(self) -> None:
        cv_path = Path("docs/Vasileios_Christopoulos_Master.md")
        sections = read_profile_sections(cv_path)
        chunks = []
        for section in sections:
            chunks.extend(chunk_profile_section(section))

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(chunk.startswith("Section:") for chunk in chunks))

    def test_high_impact_projects_chunked_per_project_entry(self) -> None:
        cv_path = Path("docs/Vasileios_Christopoulos_Master.md")
        sections = read_profile_sections(cv_path)
        project_section = next(
            section
            for section in sections
            if section.startswith("Section: High-Impact AI Projects")
        )
        chunks = chunk_profile_section(project_section)

        expected_entries = (
            "Agentic Portfolio Website",
            "AI Academic Assistant (Discord)",
            "Automated Inventory Reporting - Cizzle Brands",
            "Retail Sales Forecasting - Le Grand Cormoran",
        )
        for entry in expected_entries:
            self.assertTrue(
                any(f"Entry: {entry}" in chunk for chunk in chunks),
                f"Missing project chunk for entry '{entry}'",
            )
        self.assertTrue(
            all("Section: High-Impact AI Projects" in chunk for chunk in chunks)
        )


if __name__ == "__main__":
    unittest.main()
