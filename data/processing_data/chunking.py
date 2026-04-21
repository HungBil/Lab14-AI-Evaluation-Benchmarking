import re
from typing import List, Dict, Any

# =============================================================================
# CHUNKING TOOLS (Semantic Regex + Recursive Paragraph)
# =============================================================================
CHUNK_CHAR_SIZE = 1200 # Khoảng ~300 tokens
CHUNK_CHAR_OVERLAP = 240 # Khoảng ~60 tokens overlap

def chunk_document(text: str, filename: str) -> List[Dict[str, Any]]:
    """Tách tài liệu theo Semantic Section (===) sau đó đệ quy xuốt paragraph"""
    base_metadata = {"source": filename}
    chunks = []
    sections = re.split(r"(===.*?===)", text)
    
    current_section_title = "General"
    current_section_content = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            if current_section_content.strip():
                overlap_chunks = _split_by_paragraph(
                    current_section_content.strip(),
                    base_metadata=base_metadata,
                    section=current_section_title,
                )
                chunks.extend(overlap_chunks)
            current_section_title = part.strip("= ").strip()
            current_section_content = ""
        else:
            current_section_content += part

    if current_section_content.strip():
        overlap_chunks = _split_by_paragraph(
            current_section_content.strip(),
            base_metadata=base_metadata,
            section=current_section_title,
        )
        chunks.extend(overlap_chunks)

    return chunks

def _split_by_paragraph(text: str, base_metadata: Dict, section: str) -> List[Dict[str, Any]]:
    """Cut theo paragraph khi file quá to"""
    if len(text) <= CHUNK_CHAR_SIZE:
        return [{"text": text, "metadata": {**base_metadata, "section": section}}]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= CHUNK_CHAR_SIZE:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append({"text": current_chunk.strip(), "metadata": {**base_metadata, "section": section}})
                overlap_text = current_chunk[-CHUNK_CHAR_OVERLAP:] if len(current_chunk) > CHUNK_CHAR_OVERLAP else ""
                current_chunk = overlap_text + "\n\n" + para if overlap_text else para
            else:
                start = 0
                while start < len(para):
                    end = min(start + CHUNK_CHAR_SIZE, len(para))
                    chunks.append({"text": para[start:end].strip(), "metadata": {**base_metadata, "section": section}})
                    start = end - CHUNK_CHAR_OVERLAP
                current_chunk = ""

    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "metadata": {**base_metadata, "section": section}})

    return chunks
