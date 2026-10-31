from dataclasses import dataclass, field
from typing import List, Dict, Any
from ingestion.document_loader import DocumentPage


@dataclass
class DocumentChunk:
    """Represents an indexed chunk with text and associated source metadata."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecursiveTextSplitter:
    """
    Recursively splits text on natural boundaries (paragraphs, sentences, words)
    to maintain semantic coherence within maximum character limits and overlap.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive helper that splits text down until all parts fit chunk_size."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        # Now group splits together with overlap up to chunk_size
        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    sub_splits = self._split_text(s, new_separators)
                    final_chunks.extend(sub_splits)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merges small splits together while respecting chunk_size and chunk_overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = len(d) + (len(separator) if current_doc else 0)
            if total + _len > self.chunk_size:
                if current_doc:
                    doc_text = separator.join(current_doc).strip()
                    if doc_text:
                        docs.append(doc_text)
                    while current_doc and total > self.chunk_overlap:
                        popped = current_doc.pop(0)
                        total -= len(popped) + (len(separator) if current_doc else 0)

            current_doc.append(d)
            total = sum(len(x) for x in current_doc) + len(separator) * max(0, len(current_doc) - 1)

        if current_doc:
            doc_text = separator.join(current_doc).strip()
            if doc_text:
                docs.append(doc_text)

        return docs

    def split_documents(self, documents: List[DocumentPage]) -> List[DocumentChunk]:
        """Takes a list of DocumentPage and outputs overlapping DocumentChunk instances."""
        chunks: List[DocumentChunk] = []

        for page in documents:
            text = page.content.strip()
            if not text:
                continue

            raw_chunks = self._split_text(text, self.separators)
            doc_hash = page.metadata.get("file_hash", "doc")
            doc_name = page.metadata.get("document", "doc")
            page_num = page.metadata.get("page", 1)

            for idx, chunk_text in enumerate(raw_chunks):
                chunk_id = f"{doc_hash[:8]}_p{page_num}_c{idx + 1}"
                chunk_meta = {
                    **page.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": idx + 1,
                    "total_page_chunks": len(raw_chunks),
                }
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_meta
                ))

        return chunks
