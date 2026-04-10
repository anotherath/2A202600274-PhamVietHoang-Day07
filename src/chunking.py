from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Step 1: Split into sentences using regex
        # Pattern matches "! " or ". " or "? " or "\n"
        # We use a lookbehind to keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Step 2: Clean sentences (strip whitespace, remove empty)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Step 3: Group sentences into chunks (max_sentences_per_chunk per chunk)
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i:i + self.max_sentences_per_chunk]
            chunk = " ".join(chunk_sentences)
            chunks.append(chunk)
        
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Đoạn đã đủ ngắn thì trả về
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []
        
        # Hết separator thì cắt cứng theo chunk_size
        if not remaining_separators:
            return [current_text[i:i + self.chunk_size] 
                    for i in range(0, len(current_text), self.chunk_size)]
        
        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        
        # Separator rỗng thì cắt cứng
        if separator == "":
            return [current_text[i:i + self.chunk_size] 
                    for i in range(0, len(current_text), self.chunk_size)]
        
        # Chia bằng separator hiện tại
        parts = current_text.split(separator)
        
        result = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            # Thêm separator trở lại (trừ phần cuối)
            segment = part + (separator if i < len(parts) - 1 else "")
            
            if len(current_chunk) + len(segment) <= self.chunk_size:
                current_chunk += segment
            else:
                # Lưu chunk hiện tại nếu có
                if current_chunk:
                    result.append(current_chunk)
                
                # Segment quá dài thì chia đệ quy với separator tiếp theo
                if len(segment) > self.chunk_size:
                    sub_chunks = self._split(segment, next_separators)
                    result.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = segment
        
        # Thêm chunk cuối cùng
        if current_chunk:
            result.append(current_chunk)
        
        return result



def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # Tính tích vô hướng (dot product)
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    
    # Tính độ lớn (magnitude) của mỗi vector
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5
    
    # Trả về 0.0 nếu một trong hai vector có độ lớn bằng 0
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    # Trả về cosine similarity
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Khởi tạo các chunker có sẵn
        chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        
        results = {}
        
        for name, chunker in chunkers.items():
            chunks = chunker.chunk(text)
            
            # Tính các chỉ số thống kê
            chunk_lengths = [len(chunk) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
            
            results[name] = {
                "chunks": chunks,
                "count": len(chunks),
                "avg_length": avg_length,
                "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
                "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
            }
        
        return results