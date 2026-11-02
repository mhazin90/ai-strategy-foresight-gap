import os
import hashlib
import requests
import fitz
import pandas as pd

sources = pd.read_csv("sources.csv", dtype=str).fillna("")
os.makedirs("pdfs", exist_ok=True)
os.makedirs("data", exist_ok=True)

rows = []
for _, src in sources.iterrows():
    pdf_path = f"pdfs/{src['id']}.pdf"
    status = "ok"

    if not os.path.exists(pdf_path) and src["url"]:
        try:
            response = requests.get(src["url"], timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            if response.ok and response.content[:4] == b"%PDF":
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                status = "downloaded"
            else:
                status = f"failed_http_{response.status_code}"
        except Exception as e:
            status = "failed_" + type(e).__name__

    pages, chars, sha = 0, 0, ""
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        try:
            doc = fitz.open(pdf_path)
            pages = doc.page_count
            chars = sum(len(page.get_text()) for page in doc)
        except Exception:
            status = "unreadable"
    elif status == "ok":
        status = "missing"

    chars_per_page = chars / pages if pages else 0
    rows.append({
        "id": src["id"],
        "country": src["country"],
        "year": src["year"],
        "generation": src["generation"],
        "status": status,
        "pages": pages,
        "chars": chars,
        "chars_per_page": round(chars_per_page),
        "needs_ocr": pages > 0 and chars_per_page < 500,
        "sha256": sha,
    })

inventory = pd.DataFrame(rows)
inventory.to_csv("data/01_inventory.csv", index=False)

pd.set_option("display.width", 200)
print(inventory[["id", "status", "pages", "chars_per_page", "needs_ocr"]].to_string(index=False))
print()
print("ready:", (inventory["pages"] > 0).sum(), "/", len(inventory))
print("missing or failed:", list(inventory.loc[inventory["pages"] == 0, "id"]))
print("needs OCR:", list(inventory.loc[inventory["needs_ocr"], "id"]))
