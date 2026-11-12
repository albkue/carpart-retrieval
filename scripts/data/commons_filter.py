#!/usr/bin/env python3
"""Zero-shot CLIP relevance filter for the Commons images, plus contact sheets to review by eye.

Per image: probability that it shows the searched part (positive prompts built from the query) versus
off-topic content (diagrams, adverts, people, whole cars, engine bays, scenery). Writes relevance.csv
(file, query, score, keep) and sheets/*.jpg sorted worst first, so rejects are reviewed first.
Nothing is deleted: `keep` is only a suggestion until you have looked at the sheets.

Usage: python scripts/data/commons_filter.py --threshold 0.5
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("CARPARTS_COMMONS_DIR", REPO / "data" / "raw" / "commons"))
NEG = ["a diagram or technical drawing", "an old advertisement or poster", "a historical black and white photo",
       "a person", "a whole car on the road", "an engine bay with many parts", "a landscape or building",
       "a screenshot or text document", "an aircraft or boat", "a motorcycle or bicycle"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--per-sheet", type=int, default=30)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(a.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(a.model)
    rows = [json.loads(l) for l in (ROOT / "metadata.jsonl").open(encoding="utf-8")]
    for r in rows:  # metadata gets "file" only when a download run finishes, so rebuild the name like the collector
        ext = ".png" if r["download_url"].lower().endswith(".png") else ".jpg"
        r["file"] = str(ROOT / "images" / (hashlib.sha1(r["page_url"].encode()).hexdigest()[:16] + ext))
    rows = [r for r in rows if not r.get("duplicate") and Path(r["file"]).exists()]
    out = []
    with torch.no_grad():
        for q in sorted({r["query"] for r in rows}):
            part = q.replace(" car", "")
            texts = [f"a photo of a {part}, a car part", f"a product photo of a {part}"] + NEG
            tok = proc(text=texts, return_tensors="pt", padding=True).to(dev)
            for r in (x for x in rows if x["query"] == q):
                p = Path(r["file"])
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:  # noqa: S112 -- unreadable file is skipped, like a missing one
                    continue
                pr = model(**tok, pixel_values=proc(images=im, return_tensors="pt").to(dev)["pixel_values"]).logits_per_image.softmax(-1)[0]
                out.append({"file": str(p), "query": q, "score": float(pr[:2].sum()), "title": r["title"]})
            print(q, flush=True)
    out.sort(key=lambda r: r["score"])
    with (ROOT / "relevance.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "query", "score", "keep", "title"])
        for r in out:
            w.writerow([r["file"], r["query"], f"{r['score']:.3f}", int(r["score"] >= a.threshold), r["title"]])
    sheets = ROOT / "sheets"
    sheets.mkdir(exist_ok=True)
    for old in sheets.glob("*.jpg"):
        old.unlink()
    n, cell, cols = a.per_sheet, 220, 6
    for s in range(0, len(out), n):
        chunk = out[s:s + n]
        sheet = Image.new("RGB", (cols * cell, ((len(chunk) + cols - 1) // cols) * cell), "white")
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            im = Image.open(r["file"]).convert("RGB")
            im.thumbnail((cell - 4, cell - 28))
            x, y = (i % cols) * cell, (i // cols) * cell
            sheet.paste(im, (x + 2, y + 2))
            d.text((x + 3, y + cell - 24), f"{r['score']:.2f} {r['query']}", fill="red" if r["score"] < a.threshold else "green")
            d.text((x + 3, y + cell - 12), r["title"][5:35], fill="black")
        sheet.save(sheets / f"sheet_{s // n + 1:02d}.jpg", quality=80)
    kept = sum(r["score"] >= a.threshold for r in out)
    print(f"{len(out)} images, suggested keep {kept}, reject {len(out) - kept}; sheets in {sheets} (worst first)")


if __name__ == "__main__":
    main()
