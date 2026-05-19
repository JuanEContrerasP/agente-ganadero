"""
ingest.py — Pipeline de ingestión sin ChromaDB.

Guarda los embeddings como:
  vectorstore/embeddings.npy  → matriz float32 (N_chunks × dim)
  vectorstore/metadata.json   → lista de {texto, fuente, tipo}

Uso:
  python ingest.py
  python ingest.py --limpiar
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf
from sentence_transformers import SentenceTransformer

DATA_DIR        = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
EMBEDDINGS_FILE = VECTORSTORE_DIR / "embeddings.npy"
METADATA_FILE   = VECTORSTORE_DIR / "metadata.json"

EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE    = 350
CHUNK_OVERLAP = 60


# ── Lectores ───────────────────────────────────────────────────────────────────

def _leer_pdf(ruta: Path) -> str:
    partes = []
    with open(ruta, "rb") as f:
        for num, pag in enumerate(pypdf.PdfReader(f).pages, 1):
            txt = pag.extract_text() or ""
            if txt.strip():
                partes.append(f"--- Página {num} ---\n{txt}")
    return "\n".join(partes)


def _leer_txt(ruta: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return ruta.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"No se pudo leer {ruta.name}")


def _leer_csv(ruta: Path) -> str:
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(ruta, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"No se pudo parsear {ruta.name}")

    lineas = [f"Fuente: {ruta.name} — {len(df)} registros\n"]
    for _, fila in df.iterrows():
        partes = [
            f"{col}: {str(val).strip()}"
            for col, val in fila.items()
            if str(val).strip().lower() not in ("nan", "", "none")
        ]
        if partes:
            lineas.append(". ".join(partes) + ".")
    return "\n".join(lineas)


# ── Chunking ───────────────────────────────────────────────────────────────────

def _dividir_chunks(texto: str) -> list[str]:
    palabras = texto.split()
    chunks, inicio = [], 0
    paso = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    while inicio < len(palabras):
        fin   = min(inicio + CHUNK_SIZE, len(palabras))
        chunk = " ".join(palabras[inicio:fin]).strip()
        if chunk:
            chunks.append(chunk)
        inicio += paso
    return chunks


# ── Pipeline principal ─────────────────────────────────────────────────────────

def ingestar(limpiar: bool = False) -> None:
    VECTORSTORE_DIR.mkdir(exist_ok=True)

    # Cargar embeddings existentes (para modo incremental)
    todos_embeddings: list[list[float]] = []
    todos_meta: list[dict] = []

    if not limpiar and EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
        todos_embeddings = np.load(str(EMBEDDINGS_FILE)).tolist()
        todos_meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        fuentes_ya = {m["fuente"] for m in todos_meta}
        print(f"Modo incremental: {len(todos_meta)} chunks ya indexados.")
    else:
        fuentes_ya = set()

    print(f"Cargando modelo: {EMBED_MODEL} ...")
    modelo = SentenceTransformer(EMBED_MODEL)

    archivos = sorted(p for p in DATA_DIR.iterdir() if p.is_file())
    if not archivos:
        print("No hay archivos en data/.")
        return

    nuevos = 0
    for ruta in archivos:
        if ruta.name in fuentes_ya:
            print(f"Omitido (ya indexado): {ruta.name}")
            continue

        ext = ruta.suffix.lower()
        print(f"\nProcesando: {ruta.name}")
        try:
            if ext == ".pdf":
                texto = _leer_pdf(ruta)
            elif ext == ".txt":
                texto = _leer_txt(ruta)
            elif ext == ".csv":
                texto = _leer_csv(ruta)
            else:
                print(f"  Formato '{ext}' no soportado.")
                continue
        except Exception as exc:
            print(f"  Error: {exc}")
            continue

        if not texto.strip():
            continue

        chunks = _dividir_chunks(texto)
        print(f"  {len(chunks)} chunks.")

        # Generar embeddings en lotes
        LOTE = 64
        for i in range(0, len(chunks), LOTE):
            lote = chunks[i: i + LOTE]
            embs = modelo.encode(lote, show_progress_bar=False).tolist()
            todos_embeddings.extend(embs)
            todos_meta.extend({"texto": c, "fuente": ruta.name, "tipo": ext} for c in lote)

        nuevos += len(chunks)

    if nuevos == 0 and todos_embeddings:
        print("Nada nuevo que indexar.")
        return

    # Persistir
    np.save(str(EMBEDDINGS_FILE), np.array(todos_embeddings, dtype=np.float32))
    METADATA_FILE.write_text(json.dumps(todos_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Indexados {len(todos_meta)} chunks totales → vectorstore/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limpiar", action="store_true")
    args = parser.parse_args()
    ingestar(limpiar=args.limpiar)
