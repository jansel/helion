from __future__ import annotations

import copy
from pathlib import Path
import unittest

from plots import render_varied_attention


class VariedAttentionValidationTests(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict[str, str]]:
        return render_varied_attention._read_rows(
            Path(__file__).with_name("attention_varied_shapes_b200_750w.csv")
        )

    def test_tracked_rows_use_one_cute_version(self) -> None:
        render_varied_attention._validate_rows(self._rows())

    def test_mixed_cute_versions_are_rejected(self) -> None:
        rows = copy.deepcopy(self._rows())
        for row in rows:
            if row["implementation"] == "fa4":
                row["version"] = "fa4-v4.0.0.beta23; CuTe 4.7.0"

        with self.assertRaisesRegex(ValueError, "mix CuTe versions"):
            render_varied_attention._validate_rows(rows)

    def test_captions_are_derived_from_current_correctness(self) -> None:
        rows = self._rows()
        captions = render_varied_attention._caption_lines(rows)
        self.assertEqual(len(captions), 1)
        self.assertIn("shape 5 (SKIPPED_FULL_SHAPE_MEMORY)", captions[0])

        changed = copy.deepcopy(rows)
        for row in changed:
            row["correctness"] = "PASS"
        self.assertEqual(render_varied_attention._caption_lines(changed), [])


if __name__ == "__main__":
    unittest.main()
