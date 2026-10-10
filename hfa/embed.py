"""FashionCLIP image embeddings for every locally cached garment photo.

Reads data/interim/jpg/<accession>/*.jpg (converted RAW + cached share JPGs),
so the photo share doesn't need to be mounted.

Outputs:
  data/processed/image_embeddings.npz    accession, path, emb (n_photos x 512, float16, L2-normalized)
  data/processed/garment_embeddings.npz  accession, n_photos, emb (n_garments x 512, float32,
                                         normalized mean of the garment's photos)
Run:  .venv/bin/python -m hfa.embed [--batch 64]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
JPG = ROOT / "data/interim/jpg"
MODEL = "hf-hub:Marqo/marqo-fashionCLIP"


def load_model():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), preprocess, open_clip.get_tokenizer(MODEL), device


@torch.no_grad()
def encode_images(paths: list[Path], model, preprocess, device: str, batch: int) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch):
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch]]).to(device)
        with torch.autocast(device, enabled=device == "cuda"):
            e = model.encode_image(x).float()
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        if (i // batch) % 10 == 0:
            print(f"  {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    paths = sorted(p for p in JPG.glob("*/*.jpg") if not p.name.endswith(".part.jpg"))
    if not paths:
        raise SystemExit(f"no JPGs under {JPG}; run hfa.convert first")
    model, preprocess, _, device = load_model()
    t = time.time()
    emb = encode_images(paths, model, preprocess, device, args.batch)
    acc = np.array([p.parent.name for p in paths])
    print(f"embedded {len(paths)} photos of {len(set(acc))} garments in {time.time() - t:.0f}s on {device}")

    np.savez_compressed(ROOT / "data/processed/image_embeddings.npz", accession=acc,
                        path=np.array([str(p.relative_to(ROOT)) for p in paths]), emb=emb.astype(np.float16))
    garments = sorted(set(acc))
    g_emb = np.stack([emb[acc == a].mean(axis=0) for a in garments])
    g_emb /= np.linalg.norm(g_emb, axis=1, keepdims=True)
    np.savez_compressed(ROOT / "data/processed/garment_embeddings.npz", accession=np.array(garments),
                        n_photos=np.array([(acc == a).sum() for a in garments]), emb=g_emb.astype(np.float32))


if __name__ == "__main__":
    main()
