"""
ingest.py — Pipeline de ingestión de documentos para el Agente Ganadero.

Flujo:
  1. Lee archivos de data/ (PDF, TXT, CSV)
  2. Divide el texto en chunks con solapamiento
  3. Genera embeddings multilingües con sentence-transformers
  4. Persiste todo en ChromaDB (vectorstore/)

Uso:
  python ingest.py            # ingestión incremental (upsert)
  python ingest.py --limpiar  # borra la colección y reinicia
"""

import argparse
from pathlib import Path

import pandas as pd
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuración ──────────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
COLLECTION_NAME = "ganadero_kb"

# Modelo multilingüe: soporta español sin necesidad de traducción
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Tamaño del chunk en palabras y solapamiento entre chunks consecutivos
CHUNK_SIZE    = 350
CHUNK_OVERLAP = 60


# ── Funciones de lectura ───────────────────────────────────────────────────────

def _leer_pdf(ruta: Path) -> str:
    """Extrae el texto completo de un PDF página a página."""
    texto = []
    with open(ruta, "rb") as f:
        lector = pypdf.PdfReader(f)
        for num, pagina in enumerate(lector.pages, 1):
            contenido = pagina.extract_text() or ""
            if contenido.strip():
                texto.append(f"--- Página {num} ---\n{contenido}")
    return "\n".join(texto)


def _leer_txt(ruta: Path) -> str:
    """Lee un TXT probando distintas codificaciones comunes en Colombia."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return ruta.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"No se pudo decodificar {ruta.name} con ninguna codificación estándar.")


def _leer_csv(ruta: Path) -> str:
    """
    Convierte un CSV en texto narrativo para RAG.
    Cada fila se transforma en una oración 'columna: valor. columna: valor.'
    para que el retriever pueda encontrarla con preguntas en lenguaje natural.
    """
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(ruta, encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if df is None:
        raise ValueError(f"No se pudo leer el CSV {ruta.name}.")

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


# ── Chunking ──────────────────────────────────────────────────────────────────

def _dividir_chunks(texto: str, tamano: int = CHUNK_SIZE, solapamiento: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide texto en ventanas de 'tamano' palabras con 'solapamiento' palabras
    compartidas entre chunks adyacentes, para no perder contexto en los bordes.
    """
    palabras = texto.split()
    chunks, inicio = [], 0
    paso = max(1, tamano - solapamiento)

    while inicio < len(palabras):
        fin = min(inicio + tamano, len(palabras))
        chunk = " ".join(palabras[inicio:fin]).strip()
        if chunk:
            chunks.append(chunk)
        inicio += paso

    return chunks


# ── Pipeline principal ─────────────────────────────────────────────────────────

def ingestar(limpiar: bool = False) -> None:
    """
    Procesa todos los documentos de data/ y los indexa en ChromaDB.

    Args:
        limpiar: Si True, elimina la colección existente antes de comenzar.
    """
    # Inicializar ChromaDB con persistencia en disco
    cliente = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    if limpiar:
        try:
            cliente.delete_collection(COLLECTION_NAME)
            print("Colección anterior eliminada.")
        except Exception:
            pass

    # Cargar modelo de embeddings (se descarga automáticamente la primera vez ~120 MB)
    print(f"Cargando modelo de embeddings: {EMBED_MODEL} ...")
    modelo = SentenceTransformer(EMBED_MODEL)

    coleccion = cliente.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # similitud coseno para textos
    )

    # Recorrer todos los archivos en data/
    archivos = sorted(p for p in DATA_DIR.iterdir() if p.is_file())
    if not archivos:
        print("No hay archivos en data/. Agrega documentos y vuelve a ejecutar.")
        return

    total_chunks = 0

    for ruta in archivos:
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
                print(f"  Formato '{ext}' no soportado — omitido.")
                continue
        except Exception as exc:
            print(f"  Error leyendo {ruta.name}: {exc}")
            continue

        if not texto.strip():
            print(f"  Archivo vacío — omitido.")
            continue

        chunks = _dividir_chunks(texto)
        print(f"  {len(chunks)} chunks generados.")

        # Insertar en lotes para no sobrecargar la memoria
        LOTE = 64
        for i in range(0, len(chunks), LOTE):
            lote = chunks[i : i + LOTE]
            embeddings = modelo.encode(lote, show_progress_bar=False).tolist()
            ids = [f"{ruta.stem}__{i + j}" for j in range(len(lote))]
            metadatos = [{"fuente": ruta.name, "tipo": ext} for _ in lote]

            # upsert: actualiza si el ID ya existe, inserta si no
            coleccion.upsert(documents=lote, embeddings=embeddings, ids=ids, metadatas=metadatos)

        total_chunks += len(chunks)

    print(f"\n✓ Ingestión completa — {total_chunks} chunks almacenados en '{COLLECTION_NAME}'.")


# ── Ejecución directa ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestión de documentos para el Agente Ganadero")
    parser.add_argument("--limpiar", action="store_true", help="Borrar colección existente antes de ingestar")
    args = parser.parse_args()
    ingestar(limpiar=args.limpiar)
