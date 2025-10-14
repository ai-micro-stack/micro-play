import re
import fitz  # PyMuPDF


def chunk_a_pdf_file(file, max_chunk_size=500):
    """
    Chunk a PDF file by sentences using PyMuPdf module
    """
    sentence_enders = re.compile(r"[.!?]\s*")
    doc = fitz.open(file)
    # all_lines = []
    chunks, seq_no = [], 1
    chunk, size, setence, counter = "", 0, "", 0
    for page_num, page in enumerate(doc):
        text_blocks = page.get_text("dict")["blocks"]
        page_lines = []
        # line_no_on_page = 0
        for block in text_blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                # line_no_on_page += 1
                line_text = "".join([span["text"] for span in line["spans"]])
                page_lines.append(line_text)
        line_begin = 1
        for line_number, line_text in enumerate(page_lines):
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
