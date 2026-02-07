"""Tests for response prompt guidance and profile markdown indexing behavior."""

from pathlib import Path
import unittest

from app import SYSTEM_PROMPT
from scripts.index_profile_docs import chunk_profile_section, read_profile_sections


class PromptGuidanceTests(unittest.TestCase):
    def test_system_prompt_includes_concise_and_formatting_guidance(self) -> None:
        expected_phrases = (
            "Default to concise responses",
            "Start with a direct answer",
            "short markdown bullet points",
            "single-level bullet points",
        )
        for phrase in expected_phrases:
            self.assertIn(phrase, SYSTEM_PROMPT)


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


if __name__ == "__main__":
    unittest.main()
