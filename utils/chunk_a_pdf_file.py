import re
import fitz  # PyMuPDF
from typing import List, Dict, Any
from config import config


def chunk_a_pdf_file(file: str, max_chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Chunk a PDF file with improved strategy including overlap and semantic boundaries.

    Args:
        file: Path to the PDF file
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of chunk dictionaries
    """
    try:
        doc = fitz.open(file)
        chunks = []
        seq_no = 1

        for page_num, page in enumerate(doc):
            try:
                # Extract text with better structure preservation
                text_blocks = page.get_text("dict")["blocks"]
                page_text = _extract_page_text(text_blocks)

                if not page_text.strip():
                    continue

                # Split into paragraphs for better semantic chunking
                paragraphs = re.split(r'\n\s*\n', page_text)
                paragraphs = [p.strip() for p in paragraphs if p.strip()]

                page_chunks = _chunk_paragraphs(
                    paragraphs, file, page_num + 1, max_chunk_size, overlap, seq_no
                )
                chunks.extend(page_chunks)
                seq_no += len(page_chunks)

            except Exception as e:
                # Log error but continue with other pages
                print(f"Error processing page {page_num + 1} in {file}: {e}")
                continue

        doc.close()
        return chunks

    except Exception as e:
        # Fallback to simple method if advanced method fails
        print(f"Error in advanced PDF chunking for {file}: {e}")
        return _fallback_chunk_pdf_file(file, max_chunk_size)


def _extract_page_text(text_blocks: List[Dict]) -> str:
    """Extract and clean text from PDF text blocks."""
    page_lines = []
    for block in text_blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            line_text = "".join([span["text"] for span in line["spans"]])
            if line_text.strip():
                page_lines.append(line_text)

    return "\n".join(page_lines)


def _chunk_paragraphs(paragraphs: List[str], file: str, page_num: int,
                     max_chunk_size: int, overlap: int, start_seq: int) -> List[Dict[str, Any]]:
    """Chunk paragraphs with overlap."""
    chunks = []
    seq_no = start_seq
    current_chunk = ""
    current_size = 0
    para_start = 1

    for para_idx, paragraph in enumerate(paragraphs):
        # If adding this paragraph would exceed the limit
        if current_size + len(paragraph) > max_chunk_size and current_chunk:
            # Create chunk
            chunks.append(_create_pdf_chunk_metadata(
                file, current_chunk, seq_no, page_num, f"{para_start}-{para_idx}"
            ))
            seq_no += 1

            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:].strip()
                current_chunk = overlap_text + " " + paragraph
                current_size = len(current_chunk)
                para_start = para_idx + 1
            else:
                current_chunk = paragraph
                current_size = len(paragraph)
                para_start = para_idx + 1
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += " " + paragraph
            else:
                current_chunk = paragraph
            current_size = len(current_chunk)

    # Add final chunk
    if current_chunk:
        chunks.append(_create_pdf_chunk_metadata(
            file, current_chunk, seq_no, page_num, f"{para_start}-{len(paragraphs)}"
        ))

    return chunks


def _create_pdf_chunk_metadata(file: str, text: str, seq_no: int, page: int, line_range: str) -> Dict[str, Any]:
    """Create standardized PDF chunk metadata."""
    return {
        "id": f"f({file}):p({page})l({line_range}):{seq_no}",
        "file": file,
        "page": page,
        "line": line_range,
        "size": len(text),
        "count": len(text.split()),  # Word count approximation
        "text": text.strip(),
    }


def _fallback_chunk_pdf_file(file: str, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Fallback PDF chunking using simple sentence splitting."""
    sentence_enders = re.compile(r"[.!?]\s*")

    try:
        doc = fitz.open(file)
        chunks, seq_no = [], 1

        for page_num, page in enumerate(doc):
            text_blocks = page.get_text("dict")["blocks"]
            page_lines = []
            for block in text_blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    line_text = "".join([span["text"] for span in line["spans"]])
                    page_lines.append(line_text)

            chunk, size, sentence, counter = "", 0, "", 0
            line_begin = 1

            for line_number, line_text in enumerate(page_lines):
                sections = sentence_enders.split(line_text)
                for i, section in enumerate(sections):
                    sentence += " " + section
                    if size + len(sentence) > max_chunk_size and counter > 0:
                        line_range = f"{line_begin}-{line_number + 1}"
                        chunks.append(
                            {
                                "id": f"f({file}):p({page_num + 1})l({line_range}):{seq_no}",
                                "file": file,
                                "page": page_num + 1,
                                "line": line_range,
                                "size": size,
                                "count": counter,
                                "text": chunk.strip(),
                            }
                        )
                        seq_no += 1
                        chunk, size, line_begin, sentence, counter = (
                            sentence,
                            len(sentence),
                            line_number + 1,
                            "",
                            0,
                        )
                    else:
                        chunk += ". " + sentence
                        size += len(sentence)
                        sentence = ""
                        counter += 1

            if chunk:
                line_range = f"{line_begin}-{line_number + 1}"
                chunks.append(
                    {
                        "id": f"f({file}):p({page_num + 1})l({line_range}):{seq_no}",
                        "file": file,
                        "page": page_num + 1,
                        "line": line_range,
                        "size": size,
                        "count": counter,
                        "text": chunk.strip(),
                    }
                )

        doc.close()
        return chunks

    except Exception as e:
        print(f"Error in fallback PDF chunking for {file}: {e}")
        return []
