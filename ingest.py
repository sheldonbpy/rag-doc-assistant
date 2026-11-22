"""Step 1-2: download a focused doc set (scikit-learn User Guide), chunk by section, embed, store."""
import re, time, requests, chromadb
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

BASE = "https://scikit-learn.org/stable/"
INDEX = urljoin(BASE, "user_guide.html")
MAX_PAGES = 40          # keep it small and focused
CHUNK_CHARS = 1200
OVERLAP = 150
DB_PATH = "chroma_db"
COLLECTION = "sklearn_docs"


def get_page_urls():
    soup = BeautifulSoup(requests.get(INDEX, timeout=30).text, "html.parser")
    urls = []
    for a in soup.select("a[href]"):
        url = urldefrag(urljoin(INDEX, a["href"]))[0]
        if "/stable/modules/" in url and url.endswith(".html") and "generated" not in url and url not in urls:
            urls.append(url)
    return urls[:MAX_PAGES]


def split_long(text):
    if len(text) <= CHUNK_CHARS:
        return [text]
    parts, i = [], 0
    while i < len(text):
        parts.append(text[i:i + CHUNK_CHARS])
        i += CHUNK_CHARS - OVERLAP
    return parts


def parse_page(url):
    soup = BeautifulSoup(requests.get(url, timeout=30).text, "html.parser")
    main = soup.find("article") or soup.find("main") or soup
    for t in main(["script", "style", "nav"]):
        t.decompose()
    page_title = (main.find("h1").get_text(" ", strip=True) if main.find("h1") else url).replace("¶", "").strip()
    chunks, heading, anchor, buf = [], page_title, "", []

    def flush():
        text = re.sub(r"\s+", " ", " ".join(buf)).strip()
        if len(text) > 80:
            for part in split_long(text):
                chunks.append({"text": part, "url": url + (f"#{anchor}" if anchor else ""),
                               "title": page_title, "section": heading})

    for el in main.find_all(["h1", "h2", "h3", "p", "li", "pre"]):
        if el.name in ("h1", "h2", "h3"):
            flush(); buf = []
            heading = el.get_text(" ", strip=True).replace("¶", "").strip()
            sec = el.find_parent("section")
            anchor = sec.get("id", "") if sec else ""
        else:
            buf.append(el.get_text(" ", strip=True))
    flush()
    return chunks


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    n = 0
    for url in get_page_urls():
        chunks = parse_page(url)
        if not chunks:
            continue
        embs = model.encode([f"{c['section']}: {c['text']}" for c in chunks], normalize_embeddings=True).tolist()
        col.add(ids=[f"c{n + i}" for i in range(len(chunks))], embeddings=embs,
                documents=[c["text"] for c in chunks],
                metadatas=[{k: c[k] for k in ("url", "title", "section")} for c in chunks])
        n += len(chunks)
        print(f"{url}: {len(chunks)} chunks")
        time.sleep(0.5)  # be polite
    print(f"Done. {n} chunks stored.")


if __name__ == "__main__":
    main()
