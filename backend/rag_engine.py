import os
import re
from collections import deque
from typing import List, Optional

import pdfplumber
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv


class RAGEngine:
    def __init__(self, api_key: str = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = None
        self.all_chunks: List[Document] = []
        self.chat_history: deque = deque(maxlen=10)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4096,
            openai_api_key=self.api_key,
        )

        self.prompt_template = """
Based on the provided context and conversation history, answer the user's question in a natural, conversational way. Be concise and direct.
If the answer is truly not present anywhere in the context, say you don't have that information in the documents.
However, information may appear in different forms: references, bibliographies, tables, citations, and metadata all count as valid context. Extract data from any part of the context, including reference lists and citations.
Do not use emojis, asterisks, bold text, or markdown formatting. Just provide a clear, simple answer.
When citing information, mention the source file and page number (shown as [Source: filename, Page X] in the context).

Conversation history:
{history}

Context: {context}
Question: {question}

Answer:"""

        self.extraction_prompt_template = """
From the provided context, extract the specific information requested. Look carefully through all text including references, bibliographies, citations, tables, and metadata.
List each unique item exactly once. NEVER repeat any item you have already listed. Once you have listed all unique items, stop immediately.

Context: {context}
Question: {question}

Extracted information:"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["history", "context", "question"],
        )

        self.extraction_prompt = PromptTemplate(
            template=self.extraction_prompt_template,
            input_variables=["context", "question"],
        )

    # --- Text cleaning ---

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"^Vol\.:\([\d]+\)\s*", "", text)
        text = re.sub(r"(?m)^1 3\s*$", "", text)
        text = re.sub(r"(?m)^1 3\s+", "", text)
        text = text.replace("/g415", "ti").replace("/g410", "ff").replace("/g414", "th")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # --- PDF loading with pdfplumber (better table support) ---

    def _load_pdf(self, path: str) -> List[Document]:
        documents = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract regular text
                text = page.extract_text() or ""

                # Extract tables and format them
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    rows = []
                    for row in table:
                        cells = [str(c).strip() if c else "" for c in row]
                        rows.append(" | ".join(cells))
                    text += "\n\nTable:\n" + "\n".join(rows)

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": path, "page": i},
                        )
                    )
        return documents

    # --- Document loading ---

    def load_documents(self, file_paths: List[str]):
        documents = []
        errors = []

        for path in file_paths:
            try:
                if path.endswith(".pdf"):
                    docs = self._load_pdf(path)
                    if not docs:
                        errors.append(f"{os.path.basename(path)}: No text extracted (may be scanned/image-only)")
                    else:
                        documents.extend(docs)
                elif path.endswith(".txt"):
                    loader = TextLoader(path)
                    documents.extend(loader.load())
                else:
                    errors.append(f"{os.path.basename(path)}: Unsupported format (only PDF and TXT)")
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {str(e)}")

        if not documents:
            error_msg = "No text could be extracted from the uploaded files."
            if errors:
                error_msg += " Errors: " + "; ".join(errors)
            raise ValueError(error_msg)

        # Clean text and prepend source/page prefix
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)
            source = os.path.basename(doc.metadata.get("source", ""))
            page_num = doc.metadata.get("page", None)
            prefix = f"[Source: {source}" if source else ""
            if page_num is not None:
                prefix += f", Page {page_num + 1}"
            if prefix:
                doc.page_content = f"{prefix}] {doc.page_content}"

        # Prefer richer pages, but keep short documents if filtering would drop everything.
        filtered_documents = [doc for doc in documents if len(doc.page_content) > 100]
        if filtered_documents:
            documents = filtered_documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks were produced from the documents.")

        # Store chunks for BM25
        self.all_chunks.extend(chunks)

        # Build/update FAISS index
        if self.vector_db is None:
            self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_db.add_documents(chunks)

        self.vector_db.save_local("faiss_index")

        result = {"indexed": len(chunks), "files": len(file_paths)}
        if errors:
            result["warnings"] = errors
        return result

    # --- Hybrid retriever (FAISS + BM25) ---

    def _get_retriever(self, k: int = 12):
        faiss_retriever = self.vector_db.as_retriever(search_kwargs={"k": k})

        if self.all_chunks:
            bm25_retriever = BM25Retriever.from_documents(
                self.all_chunks, preprocess_func=self._bm25_preprocess
            )
            bm25_retriever.k = k
            return EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[0.3, 0.7],
            )
        return faiss_retriever

    @staticmethod
    def _bm25_preprocess(text: str) -> List[str]:
        # Strip parentheses so (2017) matches query for 2017
        text = re.sub(r'[().,;:\[\]]', ' ', text.lower())
        return text.split()

    def _dedupe_docs(self, docs: list) -> list:
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    # --- Chat history ---

    def _format_history(self) -> str:
        if not self.chat_history:
            return "No previous conversation."
        lines = []
        for role, text in self.chat_history:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    def clear_history(self):
        self.chat_history.clear()

    # --- Query expansion for better retrieval ---

    def _resolve_followup_query(self, question: str) -> str:
        """Expand vague follow-up turns using the previous user question."""
        q = (question or "").strip()
        if not q or not self.chat_history:
            return q

        followup_patterns = [
            r'^tell+l?\s+me\s+more\b',
            r'^(more|details)\b',
            r'^(elaborate|expand)\b',
            r'^(can you|could you)\s+(explain|expand|elaborate)\b',
            r'^(what about (that|this|it))\b',
            r'^(and (that|this|it))\b',
        ]
        is_followup = any(re.search(p, q, flags=re.IGNORECASE) for p in followup_patterns)
        if not is_followup:
            return q

        last_user_question = None
        for role, text in reversed(self.chat_history):
            if role == "user" and text.strip().lower() != q.lower():
                last_user_question = text.strip()
                break

        if not last_user_question:
            return q

        return f"{last_user_question} {q}"

    def _expand_query(self, question: str) -> str:
        # Expand year ranges like "2016 to 2020" into individual years for BM25
        range_pattern = re.compile(r'(\d{4})\s*(?:to|through|-)\s*(\d{4})')
        match = range_pattern.search(question)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if 1900 <= start <= 2100 and 1900 <= end <= 2100 and end >= start:
                years = " ".join(str(y) for y in range(start, end + 1))
                question = question + " " + years
        return question

    # --- Load vector DB if needed ---

    def _ensure_vector_db(self) -> bool:
        if self.vector_db is not None:
            return True
        if os.path.exists("faiss_index"):
            self.vector_db = FAISS.load_local(
                "faiss_index", self.embeddings, allow_dangerous_deserialization=True
            )
            # Rebuild BM25 chunks from FAISS store if needed
            if not self.all_chunks:
                self.all_chunks = list(self.vector_db.docstore._dict.values())
            return True
        return False

    # --- Query ---

    def _is_keyword_heavy(self, question: str) -> bool:
        # Detect queries that need exact keyword matching (years, author names, references)
        q = question.lower().strip()
        year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        ref_keywords = ['reference', 'cited', 'bibliography', 'author', 'citation']
        has_years = bool(year_pattern.search(q))
        has_ref = any(kw in q for kw in ref_keywords)
        if has_years and has_ref:
            return True

        # Citation-style short queries like "janssen 2017" should favor BM25.
        author_year_pattern = re.compile(r"\b[a-z][a-z\-']+\s*,?\s*(?:19|20)\d{2}\b")
        if author_year_pattern.search(q):
            return True

        # Short "name + year" questions without explicit reference keywords.
        if has_years and len(q.split()) <= 4:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'what', 'who', 'when', 'where'}
            tokens = [t for t in re.findall(r"[a-zA-Z]+", q) if len(t) >= 3]
            content_tokens = [t for t in tokens if t not in stop_words]
            if content_tokens:
                return True

        return False

    @staticmethod
    def _dedupe_answer_lines(answer: str) -> str:
        lines = answer.strip().split("\n")
        seen = set()
        unique = []
        for line in lines:
            normalized = re.sub(r'^[-\d.)\s]+', '', line).strip().lower()
            if not normalized:
                continue
            if normalized not in seen:
                seen.add(normalized)
                unique.append(line.strip())
        return "\n".join(unique)

    def _is_extraction_query(self, question: str) -> bool:
        q = question.lower()
        extraction_patterns = [
            r'\blist\b.*\bauthor',
            r'\bauthor.*\blist\b',
            r'\bgive\s+me\b.*\blist\b',
            r'\blist\b.*\breference',
            r'\bextract\b',
            r'\bnames?\s+of\b.*\bauthor',
            r'\bwho\s+(?:are|were)\b.*\bauthor',
            r'\ball\b.*\bauthor',
        ]
        return any(re.search(p, q) for p in extraction_patterns)

    def _filter_chunks_by_years(self, year_start: int, year_end: int) -> list:
        """Filter all_chunks to those containing any year in the given range."""
        year_pattern = re.compile(r'\b(' + '|'.join(str(y) for y in range(year_start, year_end + 1)) + r')\b')
        return [chunk for chunk in self.all_chunks if year_pattern.search(chunk.page_content)]

    def query(self, question: str) -> str:
        if not self._ensure_vector_db():
            return "Please upload documents first."

        retrieval_query = self._resolve_followup_query(question)
        search_query = self._expand_query(retrieval_query)
        is_extraction = self._is_extraction_query(question)

        # For extraction queries with year ranges, directly filter chunks by year
        # This avoids BM25 ranking issues where relevant chunks rank too low
        range_match = re.search(r'(\d{4})\s*(?:to|through|-)\s*(\d{4})', question)
        if is_extraction and range_match and self.all_chunks:
            year_start, year_end = int(range_match.group(1)), int(range_match.group(2))
            docs = self._filter_chunks_by_years(year_start, year_end)
        elif self._is_keyword_heavy(retrieval_query) and self.all_chunks:
            bm25 = BM25Retriever.from_documents(
                self.all_chunks, preprocess_func=self._bm25_preprocess
            )
            bm25.k = 15
            docs = self._dedupe_docs(bm25.invoke(search_query))
        else:
            retriever = self._get_retriever()
            docs = self._dedupe_docs(retriever.invoke(search_query))
        context = "\n\n".join(doc.page_content for doc in docs)

        history_text = self._format_history()
        prompt_text = self.prompt.format(
            history=history_text, context=context, question=question
        )
        response = self.llm.invoke(prompt_text)
        answer = response.content

        if is_extraction:
            answer = self._dedupe_answer_lines(answer)

        self.chat_history.append(("user", question))
        self.chat_history.append(("assistant", answer))

        return answer

    # --- Streaming query ---

    def query_stream(self, question: str):
        if not self._ensure_vector_db():
            yield "Please upload documents first."
            return

        retrieval_query = self._resolve_followup_query(question)
        search_query = self._expand_query(retrieval_query)
        is_extraction = self._is_extraction_query(question)

        range_match = re.search(r'(\d{4})\s*(?:to|through|-)\s*(\d{4})', question)
        if is_extraction and range_match and self.all_chunks:
            year_start, year_end = int(range_match.group(1)), int(range_match.group(2))
            docs = self._filter_chunks_by_years(year_start, year_end)
        elif self._is_keyword_heavy(retrieval_query) and self.all_chunks:
            bm25 = BM25Retriever.from_documents(
                self.all_chunks, preprocess_func=self._bm25_preprocess
            )
            bm25.k = 15
            docs = self._dedupe_docs(bm25.invoke(search_query))
        else:
            retriever = self._get_retriever()
            docs = self._dedupe_docs(retriever.invoke(search_query))

        context = "\n\n".join(doc.page_content for doc in docs)

        history_text = self._format_history()
        prompt_text = self.prompt.format(
            history=history_text, context=context, question=question
        )

        full_answer = ""
        for chunk in self.llm.stream(prompt_text):
            token = chunk.content
            if token:
                full_answer += token
                yield token

        self.chat_history.append(("user", question))
        self.chat_history.append(("assistant", full_answer))


rag_engine = None


def get_rag_engine(api_key: str = None):
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine(api_key)
    return rag_engine
