import re
from typing import Set

def filter_unused_citations_perplexity(text: str) -> str:
    """
    Remove unused citation entries from a block of text that ends with a
    'Citations:' list.

    Parameters
    ----------
    text : str
        The full text, containing arbitrary content followed by the literal
        separator '\n\nCitations:\n' and then one entry per line in the form
        '[number] url'.

    Returns
    -------
    str
        The same text but with the citation list filtered so it only keeps
        entries whose numeric label appears (e.g. '[12]') in the main body.
    """
    separator = "\n\nCitations:\n"
    if separator not in text:
        # If Citations section is missing, return original text directly
        return text

    main_text, citation_section = text.split(separator, 1)

    # 1) Find all citation numbers referenced in the main text
    used_numbers: Set[str] = set(re.findall(r'\[(\d+)]', main_text))

    # 2) Match "[number] URL" pattern line by line in the citation section
    kept_lines = []
    for match in re.finditer(r'^\[(\d+)]\s+(https?://\S+)', citation_section, re.MULTILINE):
        num, url = match.groups()
        if num in used_numbers:
            kept_lines.append(f"[{num}] {url}")

    # 3) Reassemble: main text + filtered citation list
    return separator.join([main_text, "\n".join(kept_lines)])


def filter_unused_citations_gemini(text: str) -> str:
    """
    Remove unused citation entries from a block of text that ends with a
    'Works cited:' list.

    Parameters
    ----------
    text : str
        The full text, containing arbitrary content followed by the literal
        separator '\n\nCitations:\n' and then one entry per line in the form
        '[number] url'.

    Returns
    -------
    str
        The same text but with the citation list filtered so it only keeps
        entries whose numeric label appears (e.g. '[12]') in the main body.
    """
    separator = "\n\n#### Works cited\n"
    if separator not in text:
        # No citations
        return text

    main_text, citation_section = text.split(separator, 1)

    # find all the cited items
    used_numbers: Set[str] = set(re.findall(r'\[(\d+)]', main_text))

    # match lines in citations
    kept_lines = []
    for line in citation_section.strip().split('\n'):
        match = re.match(r'^(\d+)\.', line)
        if match and match.group(1) in used_numbers:
            kept_lines.append(line)

    # concat the filtered citations with main text
    return separator.join([main_text, "\n".join(kept_lines)])
