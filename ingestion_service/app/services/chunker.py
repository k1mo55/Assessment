import math
import re
from dataclasses import dataclass

from ..clients.embedding_client import EmbeddingClient
from .pdf_parser import ParsedPage

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SemanticChunk:
    start_page: int
    end_page: int
    text: str


@dataclass(frozen=True)
class Sentence:
    page_number: int
    text: str


class SemanticChunker:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        similarity_threshold: float = 0.58,
        minimum_characters: int = 250,
        maximum_characters: int = 2_000,
    ) -> None:
        if minimum_characters < 1:
            raise ValueError("minimum_characters must be positive")
        if maximum_characters < minimum_characters:
            raise ValueError("maximum_characters must be at least minimum_characters")
        self._embedding_client = embedding_client
        self._similarity_threshold = similarity_threshold
        self._minimum_characters = minimum_characters
        self._maximum_characters = maximum_characters

    async def chunk(self, pages: list[ParsedPage]) -> list[SemanticChunk]:
        sentences = [
            Sentence(page.page_number, text)
            for page in pages
            for text in self._sentences(page.text)
        ]
        if not sentences:
            return []
        if len(sentences) == 1:
            sentence = sentences[0]
            return [
                SemanticChunk(
                    start_page=sentence.page_number,
                    end_page=sentence.page_number,
                    text=sentence.text,
                )
            ]

        sentence_vectors = await self._embedding_client.embed(
            [sentence.text for sentence in sentences]
        )
        chunks = await self._group_document(sentences, sentence_vectors)
        return self._merge_final_small_chunk(chunks)

    def _sentences(self, text: str) -> list[str]:
        sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
        sentences = sentences or [text.strip()]
        return [
            piece
            for sentence in sentences
            for piece in self._split_oversized_text(sentence)
        ]

    async def _group_document(
        self,
        sentences: list[Sentence],
        sentence_vectors: list[list[float]],
    ) -> list[SemanticChunk]:
        grouped: list[SemanticChunk] = []
        current = [sentences[0]]
        current_vector: list[float] | None = sentence_vectors[0]

        for index in range(1, len(sentences)):
            next_sentence = sentences[index]
            starts_new_chunk = await self._should_start_new_chunk(
                current,
                current_vector,
                next_sentence,
                sentence_vectors[index],
            )
            if starts_new_chunk:
                grouped.append(self._make_chunk(current))
                current = [next_sentence]
                current_vector = sentence_vectors[index]
            else:
                current.append(next_sentence)
                # The semantic representation must be recomputed after the
                # accumulated chunk changes; sentence-pair scores are not reused.
                current_vector = None

        grouped.append(self._make_chunk(current))
        return grouped

    async def _should_start_new_chunk(
        self,
        current: list[Sentence],
        current_vector: list[float] | None,
        next_sentence: Sentence,
        next_vector: list[float],
    ) -> bool:
        current_text = self._current_chunk_text(current)
        candidate_length = len(f"{current_text} {next_sentence.text}")
        if candidate_length > self._maximum_characters:
            return True
        if len(current_text) < self._minimum_characters:
            return False

        # Compare the next sentence with the entire rolling chunk, embedding
        # the updated chunk again whenever a sentence has been appended.
        if current_vector is None:
            current_vector = (await self._embedding_client.embed([current_text]))[0]
        similarity = _cosine_similarity(current_vector, next_vector)
        return similarity < self._similarity_threshold

    @staticmethod
    def _current_chunk_text(sentences: list[Sentence]) -> str:
        return " ".join(sentence.text for sentence in sentences)

    @staticmethod
    def _make_chunk(sentences: list[Sentence]) -> SemanticChunk:
        return SemanticChunk(
            start_page=sentences[0].page_number,
            end_page=sentences[-1].page_number,
            text=SemanticChunker._current_chunk_text(sentences),
        )

    def _split_oversized_text(self, text: str) -> list[str]:
        if len(text) <= self._maximum_characters:
            return [text]

        pieces: list[str] = []
        current = ""
        for word in text.split():
            while len(word) > self._maximum_characters:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(word[: self._maximum_characters])
                word = word[self._maximum_characters :]
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > self._maximum_characters:
                pieces.append(current)
                current = word
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    def _merge_final_small_chunk(
        self,
        chunks: list[SemanticChunk],
    ) -> list[SemanticChunk]:
        if len(chunks) < 2 or len(chunks[-1].text) >= self._minimum_characters:
            return chunks

        previous = chunks[-2]
        final = chunks[-1]
        combined_text = f"{previous.text} {final.text}"
        if len(combined_text) > self._maximum_characters:
            return chunks

        return [
            *chunks[:-2],
            SemanticChunk(
                start_page=previous.start_page,
                end_page=final.end_page,
                text=combined_text,
            ),
        ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)
