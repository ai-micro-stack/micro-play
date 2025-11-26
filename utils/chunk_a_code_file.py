import re
from typing import List, Dict, Any


def chunk_a_code_file(file: str, max_chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Chunk a code file with improved strategy that respects code structure.

    Args:
        file: Path to the code file
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of chunk dictionaries
    """
    try:
        with open(file, "r", errors="ignore") as f:
            code_text = f.read()

        # Split into logical code blocks (functions, classes, etc.)
        blocks = _split_code_into_blocks(code_text)

        chunks = []
        seq_no = 1
        current_chunk = ""
        current_size = 0
        block_start = 0

        for block_idx, block in enumerate(blocks):
            # If adding this block would exceed the limit
            if current_size + len(block) > max_chunk_size and current_chunk:
                # Create chunk
                line_range = f"{block_start + 1}-{block_idx}"
                chunks.append(_create_code_chunk_metadata(file, current_chunk, seq_no, line_range))
                seq_no += 1

                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:].strip()
                    # Find a safe break point in overlap text
                    safe_break = _find_safe_code_break(overlap_text)
                    overlap_text = overlap_text[:safe_break] if safe_break else overlap_text

                    current_chunk = overlap_text + "\n" + block
                    current_size = len(current_chunk)
                    block_start = block_idx
                else:
                    current_chunk = block
                    current_size = len(block)
                    block_start = block_idx
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n" + block
                else:
                    current_chunk = block
                current_size = len(current_chunk)

        # Add final chunk
        if current_chunk:
            line_range = f"{block_start + 1}-{len(blocks)}"
            chunks.append(_create_code_chunk_metadata(file, current_chunk, seq_no, line_range))

        return chunks

    except Exception as e:
        # Fallback to simple line-based chunking
        return _fallback_chunk_code_file(file, max_chunk_size)


def _split_code_into_blocks(code_text: str) -> List[str]:
    """Split code into logical blocks (functions, classes, etc.)."""
    # Split by common code boundaries
    lines = code_text.splitlines()
    blocks = []
    current_block = []

    for line in lines:
        stripped = line.strip()
        # Check if this is a boundary (function def, class def, etc.)
        if (stripped.startswith(('def ', 'class ', 'function ', 'public ', 'private ', 'protected '))
            or stripped.startswith(('# ', '// ', '/* '))
            or not stripped):  # Empty lines as separators

            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []

        current_block.append(line)

    if current_block:
        blocks.append('\n'.join(current_block))

    return blocks if blocks else [code_text]


def _find_safe_code_break(text: str) -> int:
    """Find a safe place to break code (after complete statements)."""
    # Look for statement endings
    patterns = [
        r'[;]\s*$',  # Semicolon
        r'[\}]\s*$',  # Closing brace
        r'[\)]\s*$',  # Closing parenthesis
    ]

    lines = text.split('\n')
    pos = 0

    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.search(pattern, line):
                return pos + len('\n'.join(lines[:i+1]))

        pos += len(line) + 1  # +1 for newline

    return len(text) // 2  # Fallback to middle


def _create_code_chunk_metadata(file: str, text: str, seq_no: int, line_range: str) -> Dict[str, Any]:
    """Create standardized code chunk metadata."""
    return {
        "id": f"f({file}):p(0)l({line_range}):{seq_no}",
        "file": file,
        "page": 0,
        "line": line_range,
        "size": len(text),
        "count": len(text.split()),  # Word count approximation
        "text": text.strip(),
    }


def _fallback_chunk_code_file(file: str, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Fallback code chunking using simple line-based approach."""
    try:
        with open(file, "r", errors="ignore") as f:
            code_text = f.read()
    except Exception:
        return []

    lines = code_text.splitlines()
    chunks, seq_no = [], 1
    chunk, size, index_begin = "", 0, 1

    for line_number, line_text in enumerate(lines):
        if size + len(line_text) > max_chunk_size:
            line_range = f"{index_begin}-{line_number}"
            chunks.append(
                {
                    "id": f"f({file}):p(0)l({line_range}):{seq_no}",
                    "file": file,
                    "page": 0,
                    "line": line_range,
                    "size": size,
                    "count": len(chunk.split()),  # Word count
                    "text": chunk.strip(),
                }
            )
            seq_no += 1
            chunk, size, index_begin = line_text, len(line_text), line_number + 1
        else:
            chunk += line_text + "\n"
            size += len(line_text) + 1

    if chunk:
        line_range = f"{index_begin}-{line_number + 1}"
        chunks.append(
            {
                "id": f"f({file}):p(0)l({line_range}):{seq_no}",
                "file": file,
                "page": 0,
                "line": line_range,
                "size": size,
                "count": len(chunk.split()),  # Word count
                "text": chunk.strip(),
            }
        )
    return chunks
