"""
rag.py — Módulo de Recuperación Aumentada (RAG).

Responsabilidades:
  - Conectarse a la colección ChromaDB ya indexada
  - Codificar la consulta del usuario con el mismo modelo de embeddings
  - Recuperar los k chunks más similares por distancia coseno
  - Devolver el contexto formateado listo para inyectar en el prompt del LLM
"""

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import VECTORSTORE_DIR, COLLECTION_NAME, EMBED_MODEL

# Número de fragmentos a recuperar por consulta
TOP_K_DEFAULT = 5


class RecuperadorRAG:
    """Encapsula toda la lógica de búsqueda semántica sobre ChromaDB."""

    def __init__(self, top_k: int = TOP_K_DEFAULT):
        self.top_k = top_k

        # Conectar a la base vectorial persistente
        self._cliente = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self._modelo   = SentenceTransformer(EMBED_MODEL)

        # Verificar que la colección exista
        colecciones = [c.name for c in self._cliente.list_collections()]
        if COLLECTION_NAME not in colecciones:
            raise RuntimeError(
                f"La colección '{COLLECTION_NAME}' no existe. "
                "Ejecuta primero: python ingest.py"
            )

        self._coleccion = self._cliente.get_collection(COLLECTION_NAME)

    # ── API pública ────────────────────────────────────────────────────────────

    def recuperar(self, consulta: str, k: int | None = None) -> list[dict]:
        """
        Devuelve los k chunks más relevantes para la consulta.

        Returns:
            Lista de dicts con claves: 'texto', 'fuente', 'distancia'
        """
        k = k or self.top_k
        embedding = self._modelo.encode([consulta]).tolist()

        resultados = self._coleccion.query(
            query_embeddings=embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            resultados["documents"][0],
            resultados["metadatas"][0],
            resultados["distances"][0],
        ):
            chunks.append({
                "texto":     doc,
                "fuente":    meta.get("fuente", "desconocida"),
                "distancia": round(float(dist), 4),
            })

        return chunks

    def construir_contexto(self, consulta: str, k: int | None = None) -> tuple[str, list[str]]:
        """
        Recupera chunks y los formatea como bloque de contexto para el LLM.

        Returns:
            (contexto_str, fuentes_únicas_ordenadas)
        """
        chunks = self.recuperar(consulta, k)

        if not chunks:
            return "No se encontró información relevante en la base de conocimiento.", []

        bloques, fuentes = [], []
        for i, c in enumerate(chunks, 1):
            bloques.append(f"[Fragmento {i} | Fuente: {c['fuente']} | Relevancia: {1 - c['distancia']:.2%}]\n{c['texto']}")
            fuentes.append(c["fuente"])

        # Deduplicar fuentes manteniendo el orden de aparición
        fuentes_unicas = list(dict.fromkeys(fuentes))

        return "\n\n".join(bloques), fuentes_unicas
