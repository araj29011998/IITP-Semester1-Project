from dotenv import load_dotenv
load_dotenv()

import re, subprocess
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "api"))
from vectordb import upsert_chunks

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "tnf"

def read_text(path: Path) -> str:
    if path.suffix.lower() in [".md",".txt",".py",".log"]:
        return path.read_text(errors="ignore")
    if path.suffix.lower() == ".pdf":
        try:
            tmp = Path(path.as_posix()+".txt")
            subprocess.run(["pdftotext","-layout",str(path),str(tmp)], check=False)
            return tmp.read_text(errors="ignore") if tmp.exists() else ""
        except Exception:
            return ""
    return ""

def chunkify(text: str, size=900, overlap=120):
    text = re.sub(r"\s+"," ", text).strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += size - overlap
    return out

def main():
    files = list(DATA_DIR.rglob("*.*"))
    chunks = []
    for f in files:
        txt = read_text(f)
        if not txt:
            continue
        for idx, ch in enumerate(chunkify(txt), start=1):
            chunks.append((ch, {"file": f.name, "page": idx}))
    if chunks:
        upsert_chunks(chunks)
        print(f"Ingested {len(chunks)} chunks from {len(files)} files.")
    else:
        print("No readable files found under data/tnf")

if __name__ == "__main__":
    main()
