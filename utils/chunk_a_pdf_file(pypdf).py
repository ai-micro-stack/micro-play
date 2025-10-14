import re
import pypdf


def chunk_a_pdf_file(file, max_chunk_size=500):
    """
    Chunk a PDF file by sentences using PyPdf module
    """
    sentence_enders = re.compile(r"[.!?]\s*")
    reader = pypdf.PdfReader(file)
    chunks, seq_no = [], 1
    chunk, size, setence, counter = "", 0, "", 0
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        source_text = page.extract_text()
        page_text = re.sub(r"\s+", " ", source_text)
        lines = page_text.splitlines()
        line_begin = 1
        for line_number, line_text in enumerate(lines):
            sections = sentence_enders.split(line_text)
            for i, section in enumerate(sections):
                setence += " " + section
                if size + len(setence) > max_chunk_size and counter > 0:
                    line_range = str(line_begin) + "-" + str(line_number + 1)
                    chunks.append(
                        {
                            "id": f"f({file}):p({page_num + 1})l({line_range}):{seq_no}",
                            "file": file,
                            "page": page_num + 1,
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
                "id": f"f({file}):p({page_num + 1})l({line_range}):{seq_no}",
                "file": file,
                "page": page_num + 1,
                "line": line_range,
                "size": size,
                "count": counter,
                "text": chunk,
            }
        )
    return chunks
