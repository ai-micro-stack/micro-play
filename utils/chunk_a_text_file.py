import re


def chunk_a_text_file(file, max_chunk_size=500):
    """
    Chunk a plain text file by sentences
    """
    sentence_enders = re.compile(r"[.!?]\s*")
    with open(file, "r", errors="ignore") as f:
        source_text = f.read()
    file_text = re.sub(r"\s+", " ", source_text)
    lines = file_text.splitlines()
    chunks, seq_no = [], 1
    chunk, size, line_begin, setence, counter = "", 0, 1, "", 0
    for line_number, line_text in enumerate(lines):
        sections = sentence_enders.split(line_text)
        for i, section in enumerate(sections):
            setence += " " + section
            if size + len(setence) > max_chunk_size and counter > 0:
                line_range = str(line_begin) + "-" + str(line_number + 1)
                chunks.append(
                    {
                        "id": f"f({file}):p(0)l({line_range}):{seq_no}",
                        "file": file,
                        "page": 0,
                        "line": line_range,
                        "size": size,
                        "count": counter,
                        "text": chunk,
                    }
                )
                seq_no += 1
                chunk, size, line_begin, setence, counter = (
                    setence,
                    len(setence),
                    line_number + 1,
                    "",
                    0,
                )
            else:
                chunk += ". " + setence
                size += len(setence)
                setence = ""
                counter += 1
    if chunk:
        line_range = str(line_begin) + "-" + str(line_number + 1)
        chunks.append(
            {
                "id": f"f({file}):p(0)l({line_range}):{seq_no}",
                "file": file,
                "page": 0,
                "line": line_range,
                "size": size,
                "count": counter,
                "text": chunk,
            }
        )
    return chunks
