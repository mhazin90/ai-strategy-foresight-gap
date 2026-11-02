import re
import pymupdf
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_WORDS = 25
MAX_WORDS = 300
STOPWORDS = set("the of and to in a is for that on with as by are be this will an it from at or which we our their its has have can not also more such these into".split())

inventory = pd.read_csv("data/01_inventory.csv")
inventory = inventory[inventory["pages"] > 0]

paragraph_rows = []
summary_rows = []
full_texts = {}

for _, doc_info in inventory.iterrows():
    doc_id = doc_info["id"]
    doc = pymupdf.open(f"pdfs/{doc_id}.pdf")

    raw_blocks = []
    for page_number, page in enumerate(doc, start=1):
        for block in page.get_text("blocks"):
            if block[6] != 0:
                continue
            text = re.sub(r"-\n(\w)", r"\1", block[4])
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                raw_blocks.append((page_number, text))

    raw_chars = sum(len(t) for _, t in raw_blocks)
    first_page_text = " ".join(t for p, t in raw_blocks if p == 1)[:150]

    block_counts = Counter(t for _, t in raw_blocks)
    clean_blocks = [(p, t) for p, t in raw_blocks if block_counts[t] < 3 and not re.fullmatch(r"[\d\W]+", t)]

    paragraphs = []
    buffer_text = ""
    buffer_page = None
    for page_number, text in clean_blocks:
        if not buffer_text:
            buffer_page = page_number
        buffer_text = (buffer_text + " " + text).strip()
        n_words = len(buffer_text.split())
        if (re.search(r"[.!?;:]\W?$", buffer_text) and n_words >= MIN_WORDS) or n_words >= MAX_WORDS:
            paragraphs.append((buffer_page, buffer_text))
            buffer_text = ""
    if len(buffer_text.split()) >= MIN_WORDS:
        paragraphs.append((buffer_page, buffer_text))

    for i, (page_number, text) in enumerate(paragraphs, start=1):
        paragraph_rows.append({
            "doc_id": doc_id,
            "country": doc_info["country"],
            "year": doc_info["year"],
            "generation": doc_info["generation"],
            "para_id": f"{doc_id}_{i:04d}",
            "page": page_number,
            "n_words": len(text.split()),
            "text": text,
        })

    all_text = " ".join(t for _, t in paragraphs)
    full_texts[doc_id] = all_text
    tokens = re.findall(r"\w+", all_text.lower())
    english_ratio = sum(w in STOPWORDS for w in tokens) / len(tokens) if tokens else 0
    kept_share = len(all_text) / raw_chars if raw_chars else 0

    summary_rows.append({
        "id": doc_id,
        "generation": doc_info["generation"],
        "paragraphs": len(paragraphs),
        "words": len(all_text.split()),
        "kept_share": round(kept_share, 2),
        "english_ratio": round(english_ratio, 2),
        "language_flag": "CHECK" if english_ratio < 0.25 else "ok",
        "first_page": first_page_text,
    })

paragraphs_df = pd.DataFrame(paragraph_rows)
summary_df = pd.DataFrame(summary_rows)
paragraphs_df.to_csv("data/02_paragraphs.csv", index=False)
summary_df.to_csv("data/02_documents.csv", index=False)

pd.set_option("display.width", 200)
print(summary_df[["id", "generation", "paragraphs", "words", "kept_share", "english_ratio", "language_flag"]].to_string(index=False))

print()
print("FIRST PAGE OF EACH DOCUMENT")
for _, row in summary_df.iterrows():
    print(f"{row['id']}: {row['first_page']}")

print()
print("DUPLICATE CHECK (similarity > 0.80)")
doc_ids = list(full_texts.keys())
tfidf = TfidfVectorizer(stop_words="english").fit_transform([full_texts[d] for d in doc_ids])
similarity = cosine_similarity(tfidf)
found = False
for i in range(len(doc_ids)):
    for j in range(i + 1, len(doc_ids)):
        if similarity[i, j] > 0.80:
            print(f"{doc_ids[i]} <-> {doc_ids[j]}: {similarity[i, j]:.2f}")
            found = True
if not found:
    print("none")

print()
print("total paragraphs:", len(paragraphs_df))
