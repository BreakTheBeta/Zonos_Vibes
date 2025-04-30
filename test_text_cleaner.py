import unittest
from TextCleaner import _base_clean_text

class TestBaseCleanText(unittest.TestCase):

    def test_normal_text(self):
        """Tests cleaning of regular text without code blocks."""
        text = "This is a normal sentence.   Another sentence follows."
        expected = "This is a normal sentence. Another sentence follows."
        self.assertEqual(_base_clean_text(text), expected)

    def test_multiline_code_block(self):
        """Tests cleaning with a multi-line code block."""
        text = "Some text before ```python\nprint('hello')\n``` and after."
        expected = "Some text before File: python print('hello'). and after."
        self.assertEqual(_base_clean_text(text), expected)

    def test_inline_code_block(self):
        """Tests cleaning with an inline code block."""
        text = "Check the `variable_name` value."
        expected = "Check the File: variable_name. value."
        self.assertEqual(_base_clean_text(text), expected)

    def test_mixed_content(self):
        """Tests cleaning with both types of code blocks and text."""
        text = "Text `inline code` then ```\nmulti-line\ncode\n``` and final text."
        expected = "Text File: inline code. then File: multi-line code. and final text."
        self.assertEqual(_base_clean_text(text), expected)

    def test_whitespace_normalization(self):
        """Tests normalization of various whitespace patterns."""
        text = "  Leading space.  Multiple   spaces   between. Trailing space.  \n Newline. \t Tab."
        expected = "Leading space. Multiple spaces between. Trailing space. Newline. Tab."
        self.assertEqual(_base_clean_text(text), expected)

    def test_empty_input(self):
        """Tests cleaning with empty input."""
        text = ""
        expected = ""
        self.assertEqual(_base_clean_text(text), expected)

    def test_whitespace_only_input(self):
        """Tests cleaning with only whitespace input."""
        text = "   \n \t  "
        expected = ""
        self.assertEqual(_base_clean_text(text), expected)

    def test_code_block_internal_whitespace(self):
        """Tests code blocks with leading/trailing whitespace inside."""
        text = "Code: ```  \n internal space \n  ``` check."
        expected = "Code: File: internal space. check."
        self.assertEqual(_base_clean_text(text), expected)

    def test_inline_code_internal_whitespace(self):
        """Tests inline code with leading/trailing whitespace inside."""
        text = "Inline: `  spaced code  ` ."
        expected = "Inline: File: spaced code. ."
        self.assertEqual(_base_clean_text(text), expected)

    def test_multiple_inline_codes(self):
        """Tests multiple inline codes in one string."""
        text = "Use `this` and `that`."
        expected = "Use File: this. and File: that.." # Corrected expectation
        self.assertEqual(_base_clean_text(text), expected)

    def test_multiple_multiline_codes(self):
        """Tests multiple multiline codes in one string."""
        text = "First ```code1``` then ```code2```."
        expected = "First File: code1. then File: code2.." # Corrected expectation
        self.assertEqual(_base_clean_text(text), expected)

if __name__ == '__main__':
    unittest.main()
