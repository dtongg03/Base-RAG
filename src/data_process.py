from pathlib import Path
import nltk
import pdfplumber
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

def load_text_files(data_dir="../data"):
    docs = []
    base = Path(data_dir)

    for path in base.rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append({
            "doc_id": path.stem,
            "text": text.strip(),
            "source": str(path),
        })
    return docs


def load_pdf_files(data_dir="../data"):
    docs = []
    base = Path(data_dir)
    for pdf_path in base.rglob("*.pdf"):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        docs.append({
            "doc_id": pdf_path.stem,
            "text": text.strip(),
            "source": str(pdf_path),
        })
    return docs

def build_sentence_chunks(docs):
    chunks = []
    for doc in docs:
        sentences = nltk.sent_tokenize(doc["text"])
        for chunk_id, sentence in enumerate(sentences):
            cleaned = sentence.strip()
            if cleaned:
                chunks.append({
                    "doc_id": doc["doc_id"],
                    "chunk_id": chunk_id,
                    "text": cleaned,
                    "source": doc["source"],
                })
    return chunks
