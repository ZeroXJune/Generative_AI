import re
import unicodedata
from typing import Tuple


class TextCleaner:
    """Clean and normalize text for embedding and processing."""

    def __init__(self, lowercase: bool = True, remove_urls: bool = True):
        """
        Initialize text cleaner.

        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls

    def clean(self, text: str) -> str:
        """
        Clean text by applying all preprocessing steps.

        Args:
            text: Raw input text

        Returns:
            Cleaned text
        """
        text = self._remove_html_tags(text)
        text = self._remove_urls(text) if self.remove_urls else text
        text = self._normalize_unicode(text)
        text = self._remove_extra_whitespace(text)
        text = self._remove_special_characters(text)
        text = text.lower() if self.lowercase else text
        return text.strip()

    @staticmethod
    def _remove_html_tags(text: str) -> str:
        """Remove HTML/XML tags."""
        return re.sub(r"<[^>]+>", "", text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove URLs."""
        return re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize unicode characters (decompose accents, etc.)."""
        return unicodedata.normalize("NFKD", text)

    @staticmethod
    def _remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespace, tabs, newlines."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _remove_special_characters(text: str) -> str:
        """Remove most special characters but keep hyphens, apostrophes."""
        text = re.sub(r"[^\w\s\-\'.]", " ", text)
        text = re.sub(r"\.{2,}", ".", text)
        return text

    def get_before_after_example(self, raw_text: str) -> dict:
        """
        Get before/after cleaning example.

        Args:
            raw_text: Raw text to clean

        Returns:
            Dictionary with before, after, and metrics
        """
        cleaned = self.clean(raw_text)
        raw_length = len(raw_text)
        cleaned_length = len(cleaned)

        return {
            "before": raw_text[:200],  # First 200 chars
            "after": cleaned[:200],
            "original_length": raw_length,
            "cleaned_length": cleaned_length,
            "reduction_pct": round(100 * (1 - cleaned_length / raw_length), 1),
        }


if __name__ == "__main__":
    cleaner = TextCleaner()

    # Example 1: HTML and special characters
    raw1 = """
    <html>
    <p>Hello!!! This is a TEST... with @#$% symbols & multiple   spaces!!!
    Check out http://example.com for more info.</p>
    </html>
    """
    example1 = cleaner.get_before_after_example(raw1)
    print("EXAMPLE 1 - HTML & Special Chars:")
    print(f"Before: {example1['before']}")
    print(f"After:  {example1['after']}")
    print(f"Reduction: {example1['reduction_pct']}%\n")

    # Example 2: Unicode and normalization
    raw2 = "Héllo wörld! Café naïve résumé..."
    example2 = cleaner.get_before_after_example(raw2)
    print("EXAMPLE 2 - Unicode Characters:")
    print(f"Before: {example2['before']}")
    print(f"After:  {example2['after']}")
    print(f"Reduction: {example2['reduction_pct']}%\n")

    # Example 3: Mixed junk
    raw3 = """
    MEETING NOTES!!!  Dr. Jane-Mary O'Connor (2026)
    ACTION ITEMS:  @@@ Please REVIEW the API docs  !!!
    https://api.example.com/v1/endpoint?param=value&other=123
    Cost:  $$$45.99 --- Too expensive???
    """
    example3 = cleaner.get_before_after_example(raw3)
    print("EXAMPLE 3 - Real-world Messy Text:")
    print(f"Before: {example3['before']}")
    print(f"After:  {example3['after']}")
    print(f"Reduction: {example3['reduction_pct']}%")
