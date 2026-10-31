import re
import unicodedata
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Cleans raw text while preserving crucial enterprise data:
    - Normalizes Unicode characters (NFKC)
    - Replaces carriage returns with standard newlines
    - Strips non-printable / control characters except standard whitespace
    - Replaces multiple spaces/tabs with a single space
    - Collapses 3+ consecutive newlines to 2 newlines (preserves paragraph boundaries)
    - Strips leading and trailing whitespace
    """
    if not text:
        return ""

    # Normalize unicode (e.g. special quotes, accented characters, dashes)
    text = unicodedata.normalize("NFKC", text)

    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printable control characters, keeping newline and tab
    text = "".join(ch for ch in text if ch in ("\n", "\t") or (unicodedata.category(ch)[0] != "C"))

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Collapse consecutive horizontal spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove trailing/leading whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse excessive blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_text_from_html(html_content: str) -> str:
    """
    Extracts readable text from HTML content:
    - Removes script, style, and metadata tags
    - Preserves semantic spacing for headers and paragraphs
    - Cleans and normalizes the resulting text
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements
    for element in soup(["script", "style", "head", "meta", "noscript", "svg"]):
        element.decompose()

    # Add spacing around block elements so text doesn't merge
    for block in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "li", "tr"]):
        block.append("\n")

    raw_text = soup.get_text(separator=" ")
    return clean_text(raw_text)
