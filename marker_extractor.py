"""
marker_extractor.py
============================
PDF parsing using marker-pdf (local open-source) with full table preservation.
Marker is the primary parser, PyMuPDF is only used as a fallback when Marker is unavailable.

Pipeline:
  PDF -> marker (local inference) -> Markdown text (preserves tables/headings)
  -> split by headings -> map to KnowledgeEntry list
"""

import re
import os
from knowledge_schema import KnowledgeEntry, KnowledgeBase, classify_knowledge_type
from logger import setup_logger

logger = setup_logger(__name__)


def parse_pdf_with_marker(pdf_path: str, max_pages: int = 0) -> str:
    """
    Parse PDF using marker-pdf (local), output Markdown string.
    Marker is always preferred for best quality results.
    PyMuPDF is only used as a fallback when Marker is unavailable.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to process, 0=all

    Returns:
        markdown_text (str): Full document as Markdown string
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        try:
            import fitz
            doc = fitz.Document(pdf_path)
            total_pages = len(doc)
            doc.close()
            pages_desc = f"first {max_pages}" if max_pages and max_pages < total_pages else f"all {total_pages}"
        except ImportError:
            total_pages = "unknown"
            pages_desc = f"up to {max_pages}" if max_pages else "all"
        
        logger.info(f"[marker] Parsing: {pdf_path} ({pages_desc} pages, local inference)...")

        model_dict = create_model_dict()
        converter = PdfConverter(artifact_dict=model_dict)

        if max_pages:
            rendered = converter(pdf_path, max_pages=max_pages)
        else:
            rendered = converter(pdf_path)

        markdown_text, _, _ = text_from_rendered(rendered)

        logger.info(f"[marker] Done. Markdown length: {len(markdown_text)} chars")
        
        if len(markdown_text.strip()) > 0:
            return markdown_text
        
        logger.warning("[WARN] Marker returned empty content, trying PyMuPDF fallback...")

    except ImportError as e:
        logger.warning(f"[WARN] marker-pdf not installed: {e}, using PyMuPDF fallback...")
    except Exception as e:
        logger.warning(f"[WARN] marker error: {e}, using PyMuPDF fallback...")

    return _fallback_pymupdf_extraction(pdf_path)


def _table_to_markdown(table) -> str:
    """Convert PyMuPDF Table object to Markdown format."""
    if table is None:
        return ""
    
    rows = table.extract()
    if not rows or len(rows) < 2:
        return ""
    
    markdown_lines = []
    
    header = rows[0]
    markdown_lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
    
    markdown_lines.append("| " + " | ".join("---" for _ in header) + " |")
    
    for row in rows[1:]:
        markdown_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    
    return "\n".join(markdown_lines)


def _fallback_pymupdf_extraction(pdf_path: str) -> str:
    """
    Fallback: when marker is unavailable, use PyMuPDF with find_tables() for table extraction.
    This is only used as a last resort when Marker fails or is not installed.
    """
    try:
        import fitz
    except ImportError:
        logger.error("[ERROR] Neither Marker nor PyMuPDF is available!")
        return ""

    doc = fitz.Document(pdf_path)
    all_blocks = []

    for page_idx, page in enumerate(doc):
        page_height = page.rect.height
        HEADER_MARGIN = 50
        FOOTER_MARGIN = 60
        
        tables = page.find_tables()
        table_rects = []
        for table in tables.tables:
            table_rects.append(table.bbox)
            table_md = _table_to_markdown(table)
            if table_md:
                all_blocks.append(f"\n\n{table_md}\n\n")

        blocks = page.get_text("blocks")

        for b in blocks:
            if b[6] != 0:
                continue
            x0, y0, x1, y1, text = b[:5]

            if y1 < HEADER_MARGIN or y0 > (page_height - FOOTER_MARGIN):
                continue

            is_in_table = False
            for table_rect in table_rects:
                if x0 >= table_rect[0] and x1 <= table_rect[2] and y0 >= table_rect[1] and y1 <= table_rect[3]:
                    is_in_table = True
                    break
            if is_in_table:
                continue

            clean_text = text.replace('\n', ' ').strip()

            if re.search(r'(\. ?){4,}\d+', clean_text):
                continue

            if clean_text:
                all_blocks.append(clean_text)

    doc.close()
    result = "\n\n".join(all_blocks)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    logger.info(f"[pymupdf] Fallback extraction complete. Length: {len(result)} chars")
    return result.strip()


def markdown_to_knowledge_entries(markdown_text: str, source_file: str = "") -> list:
    """Convert Markdown text to KnowledgeEntry list."""
    if not markdown_text or not isinstance(markdown_text, str):
        return []

    entries = []
    current_section = ""
    current_content = []

    lines = markdown_text.split('\n')
    current_heading_level = 0

    for line in lines:
        heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
        if heading_match:
            if current_section and current_content:
                content_text = '\n'.join(current_content).strip()
                if content_text:
                    entries.append(KnowledgeEntry(
                        title=current_section,
                        content=content_text,
                        source_file=source_file,
                        type=classify_knowledge_type(current_section, content_text)
                    ))
            
            current_section = heading_match.group(2).strip()
            current_content = []
            current_heading_level = len(heading_match.group(1))
        else:
            current_content.append(line)

    if current_section and current_content:
        content_text = '\n'.join(current_content).strip()
        if content_text:
            entries.append(KnowledgeEntry(
                title=current_section,
                content=content_text,
                source_file=source_file,
                content_type=classify_knowledge_type(content_text)
            ))

    if not entries and markdown_text.strip():
        entries.append(KnowledgeEntry(
            title="Document Content",
            content=markdown_text.strip(),
            source_file=source_file,
            type=classify_knowledge_type("Document Content", markdown_text.strip())
        ))

    return entries


def extract_knowledge_from_pdf(pdf_path: str, max_pages: int = 0) -> KnowledgeBase:
    """
    Main entry point: Extract knowledge entries from PDF using Marker.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to process (0 = all)

    Returns:
        KnowledgeBase containing parsed entries
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing PDF: {pdf_path}")
    logger.info(f"{'='*60}")

    if not os.path.exists(pdf_path):
        logger.error(f"[ERROR] File not found: {pdf_path}")
        return KnowledgeBase()

    markdown_text = parse_pdf_with_marker(pdf_path, max_pages)
    
    entries = markdown_to_knowledge_entries(markdown_text, os.path.basename(pdf_path))
    
    logger.info(f"\n[INFO] Extracted {len(entries)} knowledge entries")
    for i, entry in enumerate(entries[:3]):
        logger.info(f"  {i+1}. {entry.title[:40]}...")
    if len(entries) > 3:
        logger.info(f"  ... and {len(entries) - 3} more entries")

    kb = KnowledgeBase()
    kb.entries = entries
    return kb
