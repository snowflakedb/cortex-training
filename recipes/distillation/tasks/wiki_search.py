# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wiki-search: answer trivia questions with three tools over a fixed Wikipedia corpus.

The question set, corpus, tools, system prompt, and judge prompt follow
PrimeIntellect's ``wiki_search`` environment. Page titles are embedded with
ChromaDB's default local model, so the task needs no external embedding
service, and tool results are cut to ``TOOL_RESULT_MAX_CHARS`` so reading a full
page cannot overflow the context. The question dataset has a single split, so a
fixed sample of ``n_eval`` questions is held out for evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any

QUESTIONS_DATASET = "willcb/wiki-trivia-questions-v4"
CORPUS_DATASET = "willcb/rare-wiki-pages"
# Pinned so the evaluation split, the corpus, and the README results stay reproducible.
QUESTIONS_REVISION = "e89661d423c07ca26cf67d5eb311e016b7d26cd3"
CORPUS_REVISION = "12f1fb49a52f33a8e963d511b0fdf455480d8aa4"
DEFAULT_INDEX_DIR = Path.home() / ".cache" / "cortex-training" / "wiki-search-chroma"
TOOL_RESULT_MAX_CHARS = 8000

SYSTEM = (
    "Use the wiki_search_pages, wiki_view_sections, and wiki_read_section tools to "
    "answer the question. Reply with a concise final answer when confident."
)
JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the response is both correct and coherent.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only. If a response contains incoherent text, respond with "no" even if the correct answer is also present.
"""


def _tool(name: str, description: str, arg: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {arg: {"type": "string"}},
                "required": [arg],
            },
        },
    }


TOOLS = [
    _tool(
        "wiki_search_pages",
        "Search for relevant articles using title embedding similarity.",
        "query",
    ),
    _tool("wiki_view_sections", "View the sections of a page.", "page_id"),
    _tool("wiki_read_section", "Read a section of a page.", "section_id"),
]


@dataclasses.dataclass(frozen=True)
class Question:
    idx: int
    question: str
    answer: str


@dataclasses.dataclass(frozen=True)
class WikiSearchProblems:
    train: list[Question]
    eval: list[Question]


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


class WikiIndex:
    """The corpus in memory plus a title-embedding index for ``wiki_search_pages``."""

    def __init__(self, title: dict[str, str], content: dict[str, str], collection: Any):
        self.title = title
        self.content = content
        self.collection = collection

    @classmethod
    def load(cls, index_dir: str | Path = DEFAULT_INDEX_DIR) -> WikiIndex:
        import chromadb
        from datasets import load_dataset

        corpus = load_dataset(CORPUS_DATASET, split="train", revision=CORPUS_REVISION)
        title = {str(r["id"]): str(r["title"]) for r in corpus}
        content = {str(r["id"]): str(r["content"]) for r in corpus}
        # Section ids are "page_id:section"; section names can contain ":", page ids must not.
        with_colon = [page_id for page_id in title if ":" in page_id]
        if with_colon:
            raise ValueError(f"page ids must not contain ':', found {with_colon[:3]}")
        # One index per corpus revision, so a changed corpus never reuses stale embeddings.
        client = chromadb.PersistentClient(path=str(Path(index_dir) / CORPUS_REVISION))
        collection = client.get_or_create_collection(name="wiki_titles")
        if collection.count() < len(title):
            ids = list(title)
            for start in range(0, len(ids), 100):
                batch = ids[start : start + 100]
                collection.upsert(
                    ids=batch,
                    documents=[title[i] for i in batch],
                    metadatas=[{"title": title[i]} for i in batch],
                )
        return cls(title, content, collection)

    def search_pages(self, query: str) -> dict:
        results = self.collection.query(query_texts=[query], n_results=10)
        if not results or not results["metadatas"]:
            raise ValueError(f"No results found for query: {query}")
        return {
            "pages": [
                {"page_id": str(page_id), "title": str(meta["title"])}
                for page_id, meta in zip(results["ids"][0], results["metadatas"][0])
            ]
        }

    def view_sections(self, page_id: str) -> dict:
        sections = [
            {
                "section_id": f"{page_id}:{normalize_id(line.lstrip('#'))}",
                "section_name": line.lstrip("#").strip(),
            }
            for line in self.content[page_id].split("\n")
            if line.startswith("#")
        ]
        return {
            "sections": sections
            or [{"section_id": f"{page_id}:full", "section_name": "Full Page"}]
        }

    def read_section(self, section_id: str) -> str:
        if ":" not in section_id:
            raise ValueError("Invalid section_id format. Expected: page_id:section_name")
        page_id, section = section_id.split(":", 1)
        content = self.content[page_id]
        if section == "full":
            return content
        lines = content.split("\n")
        start = end = None
        for i, line in enumerate(lines):
            if not line.startswith("#"):
                continue
            if normalize_id(line.lstrip("#")) == section and start is None:
                start = i
            elif start is not None:
                end = i
                break
        if start is None:
            raise ValueError(f"Section not found: {section_id}")
        return "\n".join(lines[start : end or len(lines)])


class WikiSearchTask:
    name = "wiki-search"
    tools = TOOLS

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR, index: WikiIndex | None = None):
        self.index_dir = index_dir
        self._index = index

    @property
    def index(self) -> WikiIndex:
        if self._index is None:
            self._index = WikiIndex.load(self.index_dir)
        return self._index

    def load(self, *, seed: int, n_eval: int) -> WikiSearchProblems:
        from datasets import load_dataset

        rows = load_dataset(QUESTIONS_DATASET, split="train", revision=QUESTIONS_REVISION)
        questions = [
            Question(i, str(r["question"]), str(r["answer"])) for i, r in enumerate(rows)
        ]
        random.Random(seed).shuffle(questions)
        return WikiSearchProblems(train=questions[n_eval:], eval=questions[:n_eval])

    def user_message(self, problem: Question) -> str:
        return f"{SYSTEM}\n\nQuestion: {problem.question}"

    def call_tool(self, name: str, arguments: dict) -> str:
        """Run one tool call; errors come back as text so the agent can recover."""
        tools = {
            "wiki_search_pages": (self.index.search_pages, "query"),
            "wiki_view_sections": (self.index.view_sections, "page_id"),
            "wiki_read_section": (self.index.read_section, "section_id"),
        }
        try:
            if name not in tools:
                raise ValueError(f"Unknown tool: {name}")
            fn, arg = tools[name]
            if arg not in arguments:
                raise ValueError(f"{name} requires the argument '{arg}'")
            result = fn(str(arguments[arg]))
            text = result if isinstance(result, str) else json.dumps(result)
        except KeyError as exc:
            text = f"Error: unknown id {exc}"
        except Exception as exc:
            text = f"Error: {exc}"
        if len(text) > TOOL_RESULT_MAX_CHARS:
            text = text[:TOOL_RESULT_MAX_CHARS] + "\n...[truncated]"
        return text

    def judge_prompt(self, problem: Question, response: str) -> str:
        return JUDGE_PROMPT.format(
            question=problem.question, answer=problem.answer, response=response
        )

    def judge_verdict(self, text: str) -> float:
        return float("yes" in text.lower())
