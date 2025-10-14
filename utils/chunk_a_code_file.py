def chunk_a_code_file(file, max_chunk_size=500):
    """
    Chunk a code file by lines.
    """
    with open(file, "r", errors="ignore") as f:
        code_text = f.read()
    lines = code_text.splitlines()
    chunks, seq_no = [], 1
    chunk, size, index_begin = "", 0, 1
    for line_number, line_text in enumerate(lines):
        if size + len(line_text) > max_chunk_size:
            line_range = str(index_begin) + "-" + str(line_number)
            chunks.append(
                {
                    "id": f"f({file}):p(0)l({line_range}):{seq_no}",
                    "file": file,
                    "page": 0,
                    "line": line_range,
                    "size": size,
                    "text": chunk,
                }
            )
            seq_no += 1
            chunk, size, index_begin = line_text, len(line_text), line_number + 1
        else:
            chunk += line_text
            size += len(line_text)
    if chunk:
        line_range = str(index_begin) + "-" + str(line_number + 1)
        chunks.append(
            {
                "id": f"f({file}):p(0)l({line_range}):{seq_no}",
                "file": file,
                "page": 0,
                "line": line_range,
                "size": size,
                "text": chunk,
            }
        )
    return chunks
