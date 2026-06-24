"""
marker_extractor.py
============================
PDF parsing using marker-pdf (local open-source) with full table preservation (Markdown output).
No cloud APIs are used (per supervisor requirement).

Pipeline:
  PDF -> marker (local inference) -> Markdown text (preserves tables/headings)
  -> split by headings -> map to KnowledgeEntry list
"""

import re
import os
from knowledge_schema import KnowledgeEntry, KnowledgeBase, classify_knowledge_type


def parse_pdf_with_marker(pdf_path: str, max_pages: int = 0) -> str:
    """
    Parse PDF locally using marker-pdf, output Markdown string.
    Preserves heading hierarchy, paragraphs, and tables (as Markdown table syntax).

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to process, 0=all (warning: 180 pages may take 30+ min)

    Returns:
        markdown_text (str): Full document as Markdown string
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        # Try to get page count with fitz (PyMuPDF), but don't fail if not available
        try:
            import fitz
            doc = fitz.Document(pdf_path)
            total_pages = len(doc)
            doc.close()
            pages_desc = f"first {max_pages}" if max_pages and max_pages < total_pages else f"all {total_pages}"
        except ImportError:
            total_pages = "unknown"
            pages_desc = f"up to {max_pages}" if max_pages else "all"
        
        print(f"[marker] Parsing: {pdf_path} ({pages_desc} pages, local inference)...")
        print(f"  Note: First run downloads AI models (~1-2GB). Full 180-page processing ~30 min.")

        model_dict = create_model_dict()
        converter = PdfConverter(artifact_dict=model_dict)

        if max_pages:
            rendered = converter(pdf_path, max_pages=max_pages)
        else:
            rendered = converter(pdf_path)

        markdown_text, _, _ = text_from_rendered(rendered)

        print(f"[marker] Done. Markdown length: {len(markdown_text)} chars")
        
        # Fallback: if marker returns empty or very short content, use PyMuPDF
        if len(markdown_text.strip()) < 50:
            print(f"[WARN] Marker returned too little content ({len(markdown_text)} chars), falling back to PyMuPDF...")
            try:
                import fitz
                doc = fitz.open(pdf_path)
                fallback_text = ""
                for page in doc:
                    fallback_text += page.get_text()
                doc.close()
                if len(fallback_text.strip()) > 0:
                    markdown_text = fallback_text
                    print(f"[OK] PyMuPDF fallback: {len(markdown_text)} chars")
                else:
                    print(f"[WARN] PyMuPDF also returned empty, proceeding with empty result")
            except Exception as e:
                print(f"[WARN] PyMuPDF fallback failed: {e}")
        
        return markdown_text

    except ImportError as e:
        print(f"[WARN] marker-pdf not installed: {e}, falling back to PyMuPDF...")
        return _fallback_pymupdf_extraction(pdf_path)
    except Exception as e:
        print(f"[WARN] marker error: {e}, falling back to PyMuPDF...")
        return _fallback_pymupdf_extraction(pdf_path)


def _fallback_pymupdf_extraction(pdf_path: str) -> str:
    """
    Fallback: when marker is unavailable, use PyMuPDF coordinate-based extraction.
    Includes header/footer filtering and TOC line interception.
    """
    import fitz

    doc = fitz.Document(pdf_path)
    clean_text_blocks = []

    for page in doc:
        page_height = page.rect.height
        HEADER_MARGIN = 50
        FOOTER_MARGIN = 60
        blocks = page.get_text("blocks")

        for b in blocks:
            if b[6] != 0:
                continue
            x0, y0, x1, y1, text = b[:5]

            # Skip header/footer regions
            if y1 < HEADER_MARGIN or y0 > (page_height - FOOTER_MARGIN):
                continue

            clean_text = text.replace('\n', ' ').strip()

            # Skip TOC lines (dotted leaders followed by page numbers)
            if re.search(r'(\. ?){4,}\d+', clean_text):
                continue

            if clean_text:
                clean_text_blocks.append(clean_text)

    return "\n\n".join(clean_text_blocks)


# Plaintext heading detection for PyMuPDF output
def _detect_headings_in_plaintext(text: str) -> list[tuple[int, str]]:
    """
    Detect potential heading lines from plain text (PyMuPDF output).
    Returns: [(start_pos, heading_text), ...] sorted by position.
    """
    headings = []
    lines = text.split('\n')
    pos = 0

    for line in lines:
        stripped = line.strip()
        line_len = len(line) + 1  # +1 for \n

        if not stripped:
            pos += line_len
            continue

        is_heading = False
        clean_title = stripped

        # Rule 1: All-caps title (e.g. "TOP AND FRONT PANELS")
        if stripped.isupper() and 15 <= len(stripped) <= 80 and stripped.count(' ') >= 1:
            # Exclude address lines
            addr_keywords = ['DRIVE', 'AVENUE', 'ROAD', 'STREET', 'BOULEVARD',
                             'SAN JOSE', 'USA', 'ZIP']
            if not any(ak in stripped.upper() for ak in addr_keywords):
                is_heading = True

        # Rule 2: Chapter/section numbering (e.g. "Chapter 1", "1. Introduction")
        elif re.match(r'^(Chapter|Section|Part)\s+\d+[.:]?\s*', stripped, re.IGNORECASE):
            is_heading = True
        elif re.match(r'^\d+(\.\d+)*\s+[A-Z]', stripped):
            is_heading = True

        # Rule 3: Title-cased short line, no trailing punctuation, 3+ real words
        elif (10 <= len(stripped) <= 60
              and not re.search(r'[.!?;:]$', stripped)
              and stripped[0].isupper()):
            words = [w for w in stripped.split() if w.isalpha()]
            if len(words) >= 3:
                cap_words = [w for w in words if len(w) > 3 and w[0].isupper()]
                if len(cap_words) >= 3:
                    addr_keywords = ['drive', 'avenue', 'road', 'street', 'boulevard',
                                     'san jose', 'usa', 'zip', 'postal']
                    if not any(ak in stripped.lower() for ak in addr_keywords):
                        is_heading = True

        # Rule 4: Action/step titles (e.g. "To cable your router:", "How to reset...")
        elif re.match(r'^To\s+\w+', stripped, re.IGNORECASE) and len(stripped) < 80:
            is_heading = True
        elif re.match(r'^How\s+to\s+\w+', stripped, re.IGNORECASE) and len(stripped) < 80:
            is_heading = True

        if is_heading:
            headings.append((pos, clean_title))

        pos += line_len

    # Filter: skip cover noise in first 200 chars (address, product name, etc.)
    # But keep chapter numbering if it starts early
    filtered = []
    for pos, title in headings:
        if pos < 200:
            if re.match(r'^(Chapter|Section|Part|Appendix)\s+\d+', title, re.IGNORECASE):
                filtered.append((pos, title))
        else:
            filtered.append((pos, title))

    return filtered


def markdown_to_knowledge_entries(
    markdown_text: str,
    source_file: str,
    chunk_size: int = 800,
) -> list[KnowledgeEntry]:
    """
    Split Markdown or plain text into KnowledgeEntry list.
    Smart strategy: Use heading-based split if >= 3 headings found, otherwise paragraph-based.
    Supports:
    - Markdown (marker output, with # headings)
    - Plain text (PyMuPDF output, headings inferred from text features)
    - Academic papers (with math symbol repair and heading promotion)
    """
    entries = []

    # Try Markdown heading split first (# / ## / ###)
    heading_pattern = re.compile(r'^(#{1,3})\s+(.+)', re.MULTILINE)
    headings = list(heading_pattern.finditer(markdown_text))

    # Detect if academic paper
    is_academic = _detect_academic_paper(markdown_text)

    # Smart strategy: if >= 3 Markdown headings, use heading-based; otherwise paragraph-based
    if len(headings) >= 3:
        print(f"  [split] Using heading-based split ({len(headings)} headings found)")
        entries = _split_by_markdown_headings(
            markdown_text, headings, source_file, is_academic, chunk_size
        )
    elif headings:
        # Few Markdown headings, try plaintext detection
        plain_headings = _detect_headings_in_plaintext(markdown_text)
        if plain_headings and len(plain_headings) >= 3:
            print(f"  [split] Using plaintext heading split ({len(plain_headings)} headings found)")
            entries = _split_by_plaintext_headings(
                markdown_text, plain_headings, source_file, chunk_size
            )
        else:
            # Fall back to paragraph split
            print(f"  [split] Using paragraph-based split (only {len(headings)} Markdown headings)")
            entries = _split_by_paragraphs(
                markdown_text, source_file, is_academic, chunk_size
            )
    else:
        # No headings at all, try plaintext detection
        plain_headings = _detect_headings_in_plaintext(markdown_text)
        if plain_headings and len(plain_headings) >= 3:
            print(f"  [split] Using plaintext heading split ({len(plain_headings)} headings found)")
            entries = _split_by_plaintext_headings(
                markdown_text, plain_headings, source_file, chunk_size
            )
        else:
            # Fall back to paragraph split
            print(f"  [split] Using paragraph-based split (no headings found)")
            entries = _split_by_paragraphs(
                markdown_text, source_file, is_academic, chunk_size
            )

    return entries


def _split_by_markdown_headings(
    markdown_text: str,
    headings: list,
    source_file: str,
    is_academic: bool,
    chunk_size: int = 800,
) -> list[KnowledgeEntry]:
    """Split by Markdown headings (# / ## / ###)"""
    entries = []

    for i, match in enumerate(headings):
        title_level = len(match.group(1))
        title_text = match.group(2).strip()

        content_start = match.end()
        content_end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown_text)
        content_text = markdown_text[content_start:content_end].strip()

        if not content_text and not title_text:
            continue

        # Apply math symbol repair for academic papers
        if is_academic:
            content_text = _repair_math_symbols(content_text)

        # Heading promotion: extract real heading from content if synthetic
        use_title = title_text
        if is_academic and re.match(r'^Section\s+\d+$', title_text):
            promoted = _promote_heading_from_content(content_text)
            if promoted:
                use_title = promoted

        # Approximate page number
        try:
            import fitz
            doc = fitz.Document(source_file) if os.path.exists(source_file) else None
            total_pages = len(doc) if doc else 30
            if doc:
                doc.close()
        except:
            total_pages = 30

        approx_page = max(1, int((match.start() / len(markdown_text)) * total_pages))

        k_type = classify_knowledge_type(use_title, content_text)

        sub_entries = _smart_split(
            content_text, source_file, k_type, use_title, approx_page, chunk_size
        )
        if sub_entries:
            sub_entries[0].metadata["heading_level"] = title_level
            sub_entries[0].metadata["is_academic"] = is_academic
            entries.extend(sub_entries)
        elif use_title and len(use_title) > 3:
            entry = KnowledgeEntry(
                type=k_type,
                title=use_title,
                content="",
                source_file=os.path.basename(source_file),
                source_page=approx_page,
                keywords=_extract_keywords(use_title),
                metadata={
                    "heading_level": title_level,
                    "is_academic": is_academic,
                    "heading_promoted": use_title != title_text
                }
            )
            entries.append(entry)

    # Handle preamble content before first heading
    if headings[0].start() > 0:
        preamble = markdown_text[:headings[0].start()].strip()
        if preamble:
            preamble_entries = _smart_split(preamble, source_file, "general", "Preamble", 1)
            entries = preamble_entries + entries

    return entries


def _split_by_paragraphs(
    markdown_text: str,
    source_file: str,
    is_academic: bool,
    chunk_size: int = 800,
) -> list[KnowledgeEntry]:
    """Split by paragraphs, preserving tables as separate blocks"""
    entries = []

    paragraphs = re.split(r'\n{2,}', markdown_text)
    blocks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        is_table = '|' in para and '\n' in para

        if is_table:
            if current:
                blocks.append(current)
            blocks.append(para)
            current = ""
        elif len(para) > 200:
            if current:
                blocks.append(current)
            current = para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para

    if current:
        blocks.append(current)

    try:
        import fitz
        doc = fitz.Document(source_file) if os.path.exists(source_file) else None
        total_pages = len(doc) if doc else 30
        if doc:
            doc.close()
    except:
        total_pages = 30

    for idx, block in enumerate(blocks):
        if len(block) < 10:
            continue

        # Extract title from first line
        first_line = block.split('\n')[0].strip()
        if len(first_line) > 80:
            title = first_line[:80] + "..."
        elif first_line:
            title = first_line
        else:
            title = f"Section {idx + 1}"

        # Apply math symbol repair for academic papers
        if is_academic:
            block = _repair_math_symbols(block)
            # Heading promotion
            if re.match(r'^Section\s+\d+$', title):
                promoted = _promote_heading_from_content(block)
                if promoted:
                    title = promoted

        approx_page = max(1, int((idx / max(len(blocks), 1)) * total_pages))
        k_type = classify_knowledge_type(title, block)

        entry = KnowledgeEntry(
            type=k_type,
            title=title,
            content=block[:2000],
            source_file=os.path.basename(source_file),
            source_page=approx_page,
            keywords=_extract_keywords(title + " " + block),
            metadata={
                "has_table": '|' in block and '\n' in block,
                "is_academic": is_academic,
                "split_method": "paragraph",
                "heading_promoted": is_academic and re.match(r'^Section\s+\d+$', title) is not None
            }
        )
        entries.append(entry)

    return entries


def _is_junk_paragraph(para: str) -> bool:
    """
    Determine if a paragraph is junk/noise that should be filtered:
    - Pure image references: ![](...)
    - Figure/table captions only
    - Very short text (<15 chars without table syntax)
    """
    p = para.strip()
    if re.match(r'^!\[.*?\]\(.*?\)$', p):
        return True
    if re.match(r'^\*?\*?(Figure|Table)\s+\d+.*?\*?\*$', p) and len(p) < 80:
        return True
    if len(p) < 15 and '|' not in p:
        return True
    return False


def _detect_academic_paper(text: str) -> bool:
    """
    Detect if content appears to be an academic paper based on:
    - Presence of mathematical formulas (LaTeX or inline math)
    - Section numbering patterns (13.3.3, 2.1, etc.)
    - Academic keywords (abstract, introduction, conclusion, references)
    - Technical notation (lambda calculus, FP, FFP, etc.)
    """
    indicators = [
        r'\$\$.*?\$\$',  
        r'\$\\mu\$',     
        r'\\rho',   
        r'\\frac\{',    
        r'\\sum',       
        r'\\int',       
        r'\\sigma', 
        r'\d+(\.\d+){2,}\s+', 
        r'\b(Abstract|Introduction|Conclusion|References|Bibliography|Section|Chapter)\b',
        r'\b(FFP|FP|lambda|functional programming|metacomposition|cell|fetch)\b',
        r'<[A-Z]+,\s*[a-z]',  
        r'\$\d+\$',
        r'\^\d+\^',
    ]
    
    score = 0
    for pattern in indicators:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1
    
    return score >= 2


def _smart_split(
    text: str,
    source_file: str,
    k_type: str,
    parent_title: str,
    base_page: int,
    chunk_size: int = 800,
) -> list[KnowledgeEntry]:
    """
    Smart paragraph splitting:
    1. Filter junk paragraphs (image refs, captions, etc.)
    2. Merge adjacent short paragraphs to maintain semantic coherence
    3. Start new entry when accumulated content exceeds chunk_size
    4. Auto-detect SOP step type from content patterns
    5. Academic paper special handling: math symbol repair and heading promotion
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    # Step 1: Filter junk
    filtered = [p for p in paragraphs if not _is_junk_paragraph(p)]
    if not filtered:
        return []

    # Step 2: Merge adjacent short paragraphs
    merged_blocks = []
    current_block = ""

    for para in filtered:
        if not current_block:
            current_block = para
            continue

        is_table = '|' in para and '\n' in para
        is_long_para = len(para) > 200
        block_nearly_full = len(current_block) > chunk_size * 0.8

        if is_table or is_long_para or block_nearly_full:
            merged_blocks.append(current_block)
            current_block = para
        else:
            current_block += "\n\n" + para

    if current_block:
        merged_blocks.append(current_block)

    # Step 3: Create KnowledgeEntry objects
    entries = []
    
    # Detect if this is an academic paper
    is_academic = _detect_academic_paper(text)

    for idx, block in enumerate(merged_blocks):
        if len(block) < 20:
            continue

        # Detect SOP patterns in content
        block_type = k_type
        if re.search(r'(?:^|\n)[\s*•\-]*\d+\.', block) and parent_title.lower().startswith("to "):
            block_type = "sop_step"
        elif re.search(r'(?:^|\n)\s*-\s*\d+\.', block) and "step" in parent_title.lower():
            block_type = "sop_step"

        # Apply math symbol repair for academic papers
        if is_academic:
            block = _repair_math_symbols(block)

        # Heading promotion: extract real heading from content if synthetic
        use_title = parent_title
        if is_academic and re.match(r'^Section\s+\d+$', parent_title):
            promoted = _promote_heading_from_content(block)
            if promoted:
                use_title = promoted

        entry = KnowledgeEntry(
            type=block_type,
            title=use_title if idx == 0 else f"{use_title} (cont. {idx})",
            content=block,
            source_file=os.path.basename(source_file),
            source_page=base_page,
            keywords=_extract_keywords(block),
            metadata={
                "has_table": "|" in block and "\n" in block,
                "is_academic": is_academic,
                "heading_promoted": use_title != parent_title
            }
        )
        entries.append(entry)

    return entries


def _split_by_plaintext_headings(
    text: str,
    headings: list[tuple[int, str]],
    source_file: str,
    chunk_size: int = 800,
) -> list[KnowledgeEntry]:
    """
    Split text by detected plaintext headings into KnowledgeEntry list.

    Args:
        text: Raw plain text
        headings: [(start_pos, heading_text), ...] sorted by position
        source_file: Source file name
        chunk_size: Max chunk size
    """
    entries = []
    total_len = len(text)

    is_academic = _detect_academic_paper(text)

    for i, (pos, title) in enumerate(headings):
        content_start = pos + len(title)
        if i + 1 < len(headings):
            content_end = headings[i + 1][0]
        else:
            content_end = total_len

        content_text = text[content_start:content_end].strip()

        # Apply math symbol repair for academic papers
        if is_academic:
            content_text = _repair_math_symbols(content_text)

        # Heading promotion for academic papers
        use_title = title
        if is_academic and re.match(r'^Section\s+\d+$', title):
            promoted = _promote_heading_from_content(content_text)
            if promoted:
                use_title = promoted

        # Approximate page number by text position ratio
        approx_page = max(1, int((pos / total_len) * 180))

        k_type = classify_knowledge_type(use_title, content_text)

        sub_entries = _smart_split(
            content_text, source_file, k_type, use_title, approx_page, chunk_size
        )
        if sub_entries:
            entries.extend(sub_entries)
        elif use_title and len(use_title) > 3:
            entry = KnowledgeEntry(
                type=k_type,
                title=use_title,
                content=content_text,
                source_file=os.path.basename(source_file),
                source_page=approx_page,
                keywords=_extract_keywords(use_title + " " + content_text),
                metadata={
                    "is_academic": is_academic,
                    "heading_promoted": use_title != title
                }
            )
            entries.append(entry)

    # Handle content before first heading
    if headings[0][0] > 0:
        preamble = text[:headings[0][0]].strip()
        if preamble:
            preamble_entries = _smart_split(preamble, source_file, "general", "Preamble", 1)
            entries = preamble_entries + entries

    return entries


def _repair_math_symbols(text: str) -> str:
    """
    Post-Regex Repair for academic paper symbols:
    Fix common OCR misrecognitions of mathematical symbols in academic papers.
    The # symbol often represents μ (meaning function) in lambda calculus/FP papers.
    """
    if not text:
        return text
    
    mu_symbol = '$\\\\mu$'
    
    # Rule 0: Handle escaped \\# (LaTeX) -> μ
    text = re.sub(r'\\\\#', mu_symbol, text)
    
    # Rule 1: #( followed by expression -> μ( (meaning function application)
    text = re.sub(r'#\s*\(', mu_symbol + '(', text)
    
    # Rule 2: #e = or #x = -> μe = or μx = (meaning function applied to variable)
    text = re.sub(r'#([a-z])\s*=', mu_symbol + r'\1 =', text, flags=re.IGNORECASE)
    
    # Rule 3: # followed by angle brackets #< -> μ<
    text = re.sub(r'#\s*<', mu_symbol + '<', text)
    
    # Rule 4: #(x:y) -> μ(x:y)
    text = re.sub(r'#\s*\((\w+:\w+)\)', mu_symbol + r'(\1)', text)
    
    # Rule 5: #(<...>) -> μ(<...>)
    text = re.sub(r'#\s*\(<', mu_symbol + '(<', text)
    
    # Rule 6: pe = p#e -> pe = pμe (preserve p prefix)
    text = re.sub(r'(p)#([a-z])', r'\1' + mu_symbol + r'\2', text, flags=re.IGNORECASE)
    
    return text


def _promote_heading_from_content(content: str) -> str:
    """
    Heading Promotion: Extract real heading from content instead of synthetic "Section X".
    Scans the first 1000 chars for:
    - Numbered section patterns (13.3.3, 2.1, etc.)
    - Bold words at line start (**Summary**, **Cells**, **Fetch**)
    - Uppercase section names (CELLS, FETCHING)
    """
    if not content or len(content) < 10:
        return ""
    
    preview = content[:1000]
    
    # Rule 1: Numbered section at start (13.3.3 Summary...)
    numbered_pattern = re.match(r'^\s*(\d+(\.\d+){1,2})\s+(.{3,100})', preview)
    if numbered_pattern:
        heading = f"{numbered_pattern.group(1)} {numbered_pattern.group(3).strip()}"
        return heading[:100]
    
    # Rule 2: Bold text at start (**Summary of properties**)
    bold_pattern = re.match(r'^\s*\*\*([^*]{3,80})\*\*', preview)
    if bold_pattern:
        return bold_pattern.group(1).strip()[:100]
    
    # Rule 3: Numbered item at start (- 1) Summary...)
    item_pattern = re.match(r'^\s*[-*]\s*(\d+)\)\s+(.{3,100})', preview)
    if item_pattern:
        return f"{item_pattern.group(1)}) {item_pattern.group(2).strip()}"[:100]
    
    # Rule 4: Uppercase section name (CELLS, FETCHING AND STORING)
    upper_pattern = re.match(r'^\s*([A-Z][A-Z\s]{5,60})', preview)
    if upper_pattern:
        candidate = upper_pattern.group(1).strip()
        if len(candidate) >= 5 and candidate.count(' ') >= 1:
            return candidate[:100]
    
    # Rule 5: First line as heading (before first newline)
    first_line = re.match(r'^\s*([^\n]{10,120})', preview)
    if first_line:
        return first_line.group(1).strip()[:100]
    
    return ""


def _extract_keywords(text: str, max_kw: int = 8) -> list[str]:
    """
    Rule-based keyword extraction (no NLP library required):
    - Extract uppercase words >= 3 chars (proper nouns, acronyms)
    - Extract important lowercase words based on frequency
    - Extract step/phase patterns
    - Extract quoted terms
    """
    keywords = []
    
    # Remove noise characters
    clean_text = re.sub(r'[<>{}[\]\\|]', ' ', text)
    
    # 1. Uppercase words (technical terms, abbreviations) - at least 3 chars
    upper_words = re.findall(r'\b[A-Z]{3,}\b', clean_text)
    keywords.extend(list(set(upper_words))[:4])
    
    # 2. Title/topic words - first words of sentences or after headings
    title_words = re.findall(r'(?:^|\n|\.\s|\:\s)([A-Z][a-zA-Z]{2,})', clean_text)
    keywords.extend([w for w in title_words if w.lower() not in {'the', 'and', 'for', 'are', 'you', 'your', 'this', 'that', 'with', 'from', 'into', 'onto', 'over', 'under', 'through', 'between', 'among', 'without', 'within', 'upon', 'than', 'but', 'or', 'so', 'yet', 'because', 'while', 'if', 'when', 'though', 'although', 'unless', 'until', 'since', 'as', 'than', 'that', 'which', 'who', 'whom', 'whose', 'what', 'where', 'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'or', 'and', 'but', 'if', 'because', 'while', 'although', 'though', 'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}][:4])
    
    # 3. "Step X" / "Phase X" / "Chapter X" patterns
    step_patterns = re.findall(r'\b(Step\s+\d+|Phase\s+\d+|Chapter\s+\d+|Section\s+\d+)\b', clean_text, re.IGNORECASE)
    keywords.extend(step_patterns[:2])
    
    # 4. Quoted terms
    quoted = re.findall(r'"([^"]{3,30})"', clean_text)
    keywords.extend(quoted[:2])
    
    # 5. Technical terms from common tech vocab
    tech_terms = re.findall(r'\b(printer|scanner|document|network|wireless|USB|Wi-Fi|Ethernet|software|driver|settings|paper|cartridge|ink|toner|control|panel|display|button|menu|option|feature|function|system|device|installation|setup|configuration|connection|interface|memory|storage|data|file|folder|image|text|table|page|format|size|quality|speed|performance|support|help|guide|manual|troubleshoot|error|problem|solution|warning|caution|note)\b', clean_text, re.IGNORECASE)
    keywords.extend([w.lower() for w in tech_terms][:3])

    # Deduplicate, filter short words, and truncate
    seen = set()
    result = []
    for kw in keywords:
        kw_clean = kw.strip()
        # Filter out single letters and common words
        if len(kw_clean) >= 3 and kw_clean.lower() not in {'the', 'and', 'for', 'are', 'you', 'your', 'this', 'that', 'with', 'from', 'into', 'onto', 'over', 'under', 'through', 'between', 'among', 'without', 'within', 'upon', 'than', 'but', 'or', 'so', 'yet', 'because', 'while', 'if', 'when', 'though', 'although', 'unless', 'until', 'since', 'as'}:
            kw_normalized = kw_clean.lower()
            if kw_normalized not in seen:
                seen.add(kw_normalized)
                result.append(kw_clean)
        if len(result) >= max_kw:
            break

    return result


# Main entry point: one-step PDF -> KnowledgeBase
def extract_knowledge_from_pdf(pdf_path: str) -> KnowledgeBase:
    """
    Full pipeline: PDF file -> KnowledgeBase (with all structured KnowledgeEntry objects).
    """
    print(f"\n{'='*50}")
    print(f"  Extracting structured knowledge from: {pdf_path}")
    print(f"{'='*50}\n")

    # Step 1: Parse PDF -> Markdown
    markdown_text = parse_pdf_with_marker(pdf_path)

    # Step 2: Markdown -> structured knowledge entries
    print("\n  Splitting into structured knowledge entries...")
    entries = markdown_to_knowledge_entries(markdown_text, source_file=pdf_path)

    # Step 3: Assemble KnowledgeBase
    kb = KnowledgeBase(source_file=os.path.basename(pdf_path))
    for entry in entries:
        kb.add(entry)

    print(f"\n  Extraction complete.")
    print(f"  Total entries: {len(kb.entries)}")

    type_counts = {}
    for e in kb.entries:
        type_counts[e.type] = type_counts.get(e.type, 0) + 1
    print(f"  Type distribution: {type_counts}")

    return kb


if __name__ == "__main__":
    import sys

    pdf_file = "sample_manual.pdf"
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]

    if not os.path.exists(pdf_file):
        print(f"[ERROR] File not found: {pdf_file}")
        sys.exit(1)

    kb = extract_knowledge_from_pdf(pdf_file)

    # Preview first 3 entries
    print("\n  First 3 entries preview:")
    for i, entry in enumerate(kb.entries[:3]):
        print(f"\n  --- Entry {i+1} ---")
        print(f"  Type: {entry.type}")
        print(f"  Title: {entry.title}")
        print(f"  Content: {entry.content[:150]}...")
        print(f"  Keywords: {entry.keywords}")

    kb.save_json("knowledge_output_test.json")
