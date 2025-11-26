import re
from typing import List, Dict, Any
from config import config


def chunk_a_text_file(file: str, max_chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Chunk a plain text file with improved strategy including overlap and semantic boundaries.

    Args:
        file: Path to the text file
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of chunk dictionaries
    """
    try:
        with open(file, "r", errors="ignore") as f:
            source_text = f.read()

        # Normalize whitespace
        file_text = re.sub(r"\s+", " ", source_text.strip())

        # Split into paragraphs first (better semantic boundaries)
        paragraphs = re.split(r'\n\s*\n', file_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        seq_no = 1
        current_chunk = ""
        current_size = 0

        for para_idx, paragraph in enumerate(paragraphs):
            # If adding this paragraph would exceed the limit
            if current_size + len(paragraph) > max_chunk_size and current_chunk:
                # Create chunk
                chunks.append(_create_chunk_metadata(file, current_chunk, seq_no, 0, para_idx))
                seq_no += 1

                # Start new chunk with overlap from previous chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:].strip()
                    current_chunk = overlap_text + " " + paragraph
                    current_size = len(current_chunk)
                else:
                    current_chunk = paragraph
                    current_size = len(paragraph)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
                current_size = len(current_chunk)

        # Add final chunk
        if current_chunk:
            chunks.append(_create_chunk_metadata(file, current_chunk, seq_no, 0, len(paragraphs)))

        return chunks

    except Exception as e:
        # Fallback to simple line-based chunking if advanced method fails
        return _fallback_chunk_text_file(file, max_chunk_size)


def _create_chunk_metadata(file: str, text: str, seq_no: int, page: int, line_info: Any) -> Dict[str, Any]:
    """Create standardized chunk metadata."""
    return {
        "id": f"f({file}):p({page})l({line_info}):{seq_no}",
        "file": file,
        "page": page,
        "line": str(line_info),
        "size": len(text),
        "count": len(text.split()),  # Word count approximation
        "text": text.strip(),
    }


def _fallback_chunk_text_file(file: str, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Fallback chunking method using simple sentence splitting."""
    sentence_enders = re.compile(r"[.!?]\s*")

    try:
        with open(file, "r", errors="ignore") as f:
            source_text = f.read()
    except Exception:
        return []

    file_text = re.sub(r"\s+", " ", source_text)
    lines = file_text.splitlines()
    chunks, seq_no = [], 1
    chunk, size, line_begin, sentence, counter = "", 0, 1, "", 0

    for line_number, line_text in enumerate(lines):
        sections = sentence_enders.split(line_text)
        for i, section in enumerate(sections):
            sentence += " " + section
            if size + len(sentence) > max_chunk_size and counter > 0:
                line_range = f"{line_begin}-{line_number + 1}"
                chunks.append(
                    {
                        "id": f"f({file}):p(0)l({line_range}):{seq_no}",
                        "file": file,
                        "page": 0,
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
                "id": f"f({file}):p(0)l({line_range}):{seq_no}",
                "file": file,
                "page": 0,
                "line": line_range,
                "size": size,
                "count": counter,
                "text": chunk.strip(),
            }
        )
    return chunks
