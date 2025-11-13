import argparse, os, json, re, csv, math, heapq, pickle, time
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# ========== Utils ==========
_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[^\w]+", re.UNICODE)

RU_STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под тебя ихних при откуда почему куда зачем всех никогда можно нельзя каждом любом
""".split())

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.replace("\u00A0", " ")
    s = _WS.sub(" ", s)
    return s.strip()

def tokenize(s: str) -> List[str]:
    s = s.lower()
    s = _ALNUM.sub(" ", s)
    toks = [t for t in s.split() if t and t not in RU_STOP]
    return toks

def sliding_windows(text: str, size: int, overlap: int) -> List[Tuple[int,int,str]]:
    if not text: return []
    assert size > 0
    step = max(1, size - overlap)
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + size)
        # расширяем до ближайшей границы по пробелу
        if j < n:
            k = text.rfind(" ", i, j)
            if k != -1 and k > i + size*0.6:
                j = k
        chunk = text[i:j].strip()
        if chunk:
            out.append((i, j, chunk))
        i += step
    return out

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def minmax_scale(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12: return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# ========== BM25 (простая реализация + сериализация) ==========
@dataclass
class BM25Index:
    doc_tokens: List[List[str]] = field(default_factory=list)
    idf: Dict[str, float] = field(default_factory=dict)
    avgdl: float = 0.0
    k1: float = 1.4
    b: float = 0.35
    vocab_df: Dict[str, int] = field(default_factory=dict)
    # не сохраняем целиком postings (экономим память), считаем tf на лету из doc_tokens

    def build(self, docs_tokens: List[List[str]]):
        self.doc_tokens = docs_tokens
        N = len(docs_tokens)
        df = defaultdict(int)
        for toks in docs_tokens:
            for t in set(toks):
                df[t] += 1
        self.vocab_df = dict(df)
        self.idf = {t: math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0) for t, df_t in df.items()}
        self.avgdl = np.mean([len(toks) for toks in docs_tokens]) if N else 0.0

    def score(self, q_toks: List[str], web_idx: int) -> float:
        toks = self.doc_tokens[web_idx]
        dl = len(toks) + 1e-9
        tf = Counter(toks)
        score = 0.0
        for t in q_toks:
            if t not in self.idf: 
                continue
            f = tf.get(t, 0)
            if f == 0: 
                continue
            idf = self.idf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
            score += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
        return score

    def top_k(self, q_toks: List[str], k: int) -> List[Tuple[int, float]]:
        # линейный проход (для тысяч/десятков тысяч чанков норм)
        scores = []
        for i in range(len(self.doc_tokens)):
            s = self.score(q_toks, i)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "doc_tokens": self.doc_tokens,
                "idf": self.idf,
                "avgdl": self.avgdl,
                "k1": self.k1,
                "b": self.b,
                "vocab_df": self.vocab_df
            }, f)

    @staticmethod
    def load(path: str) -> "BM25Index":
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx = BM25Index()
        idx.doc_tokens = d["doc_tokens"]
        idx.idf = d["idf"]
        idx.avgdl = d["avgdl"]
        idx.k1 = d["k1"]
        idx.b = d["b"]
        idx.vocab_df = d["vocab_df"]
        return idx

# ========== Embeddings / ANN ==========
def get_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("Install sentence-transformers: pip install sentence-transformers") from e
    m = SentenceTransformer(model_name, device="cpu")
    return m

def encode_texts(embedder, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return vecs.astype(np.float32)

class ANN:
    def __init__(self, dim: int):
        self.dim = dim
        self.use_hnsw = False
        self.index = None
        self.vecs = None

    def build(self, vecs: np.ndarray, M: int = 64, efC: int = 300):
        # пытаемся hnswlib
        try:
            import hnswlib
            self.index = hnswlib.Index(space='cosine', dim=self.dim)
            self.index.init_index(max_elements=vecs.shape[0], ef_construction=efC, M=M)
            self.index.add_items(vecs, np.arange(vecs.shape[0]))
            self.index.set_ef(200)
            self.use_hnsw = True
        except Exception:
            self.vecs = vecs
            self.use_hnsw = False

    def query(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_hnsw:
            labels, dists = self.index.knn_query(q, k=k)
            # hnswlib cosine gives smaller distance for closer; convert to similarity ~ 1 - dist
            sims = 1.0 - dists
            return labels, sims
        else:
            # brute-force dot product on normalized vectors = cosine
            sims = self.vecs @ q.T  # [N, B]
            # for single vector q: shape [N, 1]
            if sims.ndim == 2 and sims.shape[1] == 1:
                sims = sims[:, 0]
                idx = np.argpartition(-sims, np.arange(min(k, sims.shape[0])))[:k]
                idx_sorted = idx[np.argsort(-sims[idx])]
                return idx_sorted.reshape(1, -1), sims[idx_sorted].reshape(1, -1)
            else:
                raise ValueError("Batch queries not supported in brute force mode")

# ========== 1) BUILD-INDEX ==========
def cmd_build_index(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.corpus)
    needed = {"web_id","title","url","kind","text"}
    assert needed.issubset(set(df.columns)), f"CSV must have columns: {needed}"
    chunks = []
    bm25_docs_tokens = []

    h1_size, h1_ov = args.h1_size, args.h1_overlap
    h2_size, h2_ov = args.h2_size, args.h2_overlap

    print(f"[build] reading {len(df)} docs, chunking H1={h1_size}/{h1_ov}, H2={h2_size}/{h2_ov}")
    for _, row in df.iterrows():
        web_id = str(row["web_id"])
        title = normalize_text(str(row["title"]))
        url = normalize_text(str(row["url"]))
        typ = str(row["kind"])
        text = normalize_text(str(row["text"]))

        for idx, (i, j, chunk) in enumerate(sliding_windows(text, h1_size, h1_ov)):
            chunk_id = f"{web_id}#h1#{idx:05d}"
            chunks.append({"chunk_id":chunk_id,"web_id":web_id,"title":title,"url":url,"type":typ,"text":chunk,"level":"h1"})
            
        for idx, (i, j, chunk) in enumerate(sliding_windows(text, h2_size, h2_ov)):
            chunk_id = f"{web_id}#h2#{idx:05d}"
            chunks.append({"chunk_id":chunk_id,"web_id":web_id,"title":title,"url":url,"type":typ,"text":chunk,"level":"h2"})

    chunks_path = os.path.join(args.out_dir, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"[build] chunks saved: {chunks_path} ({len(chunks)} chunks)")

    for ch in chunks:
        toks = tokenize(((ch["title"]+" ") * 3) + ((ch["url"]+" ") * 2) + ch["text"])
        bm25_docs_tokens.append(toks)
    bm25 = BM25Index(k1=args.k1, b=args.b)
    bm25.build(bm25_docs_tokens)
    bm25_path = os.path.join(args.out_dir, "bm25_index.pkl")
    bm25.save(bm25_path)
    print(f"[build] bm25 index saved: {bm25_path} (avgdl={bm25.avgdl:.2f}, vocab={len(bm25.idf)})")

    # Embeddings for H1/H2 separately
    embedder = get_embedder(args.embed_model)
    texts_h1 = [ch["text"] for ch in chunks if ch["level"]=="h1"]
    texts_h2 = [ch["text"] for ch in chunks if ch["level"]=="h2"]
    print(f"[build] encoding H1 ({len(texts_h1)})")
    emb_h1 = encode_texts(embedder, texts_h1, batch_size=args.batch)
    print(f"[build] encoding H2 ({len(texts_h2)})")
    emb_h2 = encode_texts(embedder, texts_h2, batch_size=args.batch)
    np.save(os.path.join(args.out_dir, "emb_h1.npy"), emb_h1)
    np.save(os.path.join(args.out_dir, "emb_h2.npy"), emb_h2)

    # maps
    idx_h1, idx_h2 = {}, {}
    i1 = 0
    for ch in chunks:
        if ch["level"]=="h1":
            idx_h1[ch["chunk_id"]] = i1
            i1 += 1
    i2 = 0
    for ch in chunks:
        if ch["level"]=="h2":
            idx_h2[ch["chunk_id"]] = i2
            i2 += 1
    with open(os.path.join(args.out_dir, "idx_h1.json"), "w", encoding="utf-8") as f:
        json.dump(idx_h1, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "idx_h2.json"), "w", encoding="utf-8") as f:
        json.dump(idx_h2, f, ensure_ascii=False)

# ========== 2) RETRIEVE ==========
def load_chunks(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def rrf_fusion(rank_dicts: List[Dict[str,int]], k: int = 60) -> Dict[str, float]:
    scores = defaultdict(float)
    for rnk in rank_dicts:
        for cid, r in rnk.items():
            scores[cid] += 1.0 / (k + r)
    return scores

def rank_list(ids: List[str], scores: List[float]) -> Dict[str,int]:
    order = np.argsort(-np.array(scores))
    return {ids[i]: int(pos+1) for pos, i in enumerate(order)}

def build_ann_from_disk(out_dir: str, level: str):
    emb = np.load(os.path.join(out_dir, f"emb_{level}.npy"))
    ann = ANN(dim=emb.shape[1]); ann.build(emb, M=64, efC=300)
    return ann, emb

def cmd_retrieve(args):
    out_dir = args.index_dir
    chunks = load_chunks(os.path.join(out_dir, "chunks.jsonl"))
    bm25 = BM25Index.load(os.path.join(out_dir, "bm25_index.pkl"))

    chunk_ids_h1 = [ch["chunk_id"] for ch in chunks if ch["level"]=="h1"]
    chunk_ids_h2 = [ch["chunk_id"] for ch in chunks if ch["level"]=="h2"]
    title_by_chunk = {ch["chunk_id"]: ch["title"] for ch in chunks}
    doc_by_chunk = {ch["chunk_id"]: ch["web_id"] for ch in chunks}

    ann_h1, emb_h1 = build_ann_from_disk(out_dir, "h1")
    ann_h2, emb_h2 = build_ann_from_disk(out_dir, "h2")

    embedder = get_embedder(args.embed_model)

    qs = pd.read_csv(args.questions)
    rows_out = []
    print(f"[retrieve] queries: {len(qs)}")
    for _, row in qs.iterrows():
        qid = row["q_id"]
        query = normalize_text(str(row["query"]))
        q_tok = tokenize(query)

        all_chunk_ids = [ch["chunk_id"] for ch in chunks]
        top_lex = bm25.top_k(q_tok, k=args.top_lex)
        lex_ids = [all_chunk_ids[i] for i, s in top_lex]
        lex_scores = [s for i, s in top_lex]
        rank_lex = rank_list(lex_ids, lex_scores)

        # --- Dense H1 ---
        q_vec = encode_texts(embedder, [query])[0].reshape(1, -1).astype(np.float32)
        labels_h1, sims_h1 = ann_h1.query(q_vec, k=min(args.top_dense//2, len(chunk_ids_h1)))
        ids_h1 = [chunk_ids_h1[i] for i in labels_h1[0]]
        sc_h1 = sims_h1[0].tolist()
        rank_h1 = rank_list(ids_h1, sc_h1)

        labels_h2, sims_h2 = ann_h2.query(q_vec, k=min(args.top_dense//2, len(chunk_ids_h2)))
        ids_h2 = [chunk_ids_h2[i] for i in labels_h2[0]]
        sc_h2 = sims_h2[0].tolist()
        rank_h2 = rank_list(ids_h2, sc_h2)

        rrf = rrf_fusion([rank_lex, rank_h1, rank_h2], k=args.rrf_k)

        cand_ids = set(lex_ids) | set(ids_h1) | set(ids_h2)

        cand_sorted = sorted(cand_ids, key=lambda cid: rrf.get(cid, 0.0), reverse=True)[:args.pool_size]

        bm25_map = {cid: 0.0 for cid in cand_sorted}
        dense_h1_map = {cid: 0.0 for cid in cand_sorted}
        dense_h2_map = {cid: 0.0 for cid in cand_sorted}
        rrf_map = {cid: rrf.get(cid, 0.0) for cid in cand_sorted}

        lex_map = {cid: sc for cid, sc in zip(lex_ids, lex_scores)}
        for cid in cand_sorted:
            bm25_map[cid] = float(lex_map.get(cid, 0.0))

        h1_map = {cid: sc for cid, sc in zip(ids_h1, sc_h1)}
        h2_map = {cid: sc for cid, sc in zip(ids_h2, sc_h2)}
        for cid in cand_sorted:
            dense_h1_map[cid] = float(h1_map.get(cid, 0.0))
            dense_h2_map[cid] = float(h2_map.get(cid, 0.0))

        # title hit
        q_tokens_set = set(q_tok)
        for cid in cand_sorted:
            title_toks = set(tokenize(title_by_chunk.get(cid, "")))
            title_hit = 1 if (q_tokens_set & title_toks) else 0
            rows_out.append({
                "q_id": qid,
                "chunk_id": cid,
                "web_id": doc_by_chunk[cid],
                "bm25": bm25_map[cid],
                "dense_h1": dense_h1_map[cid],
                "dense_h2": dense_h2_map[cid],
                "rrf": rrf_map[cid],
                "title_hit": title_hit
            })

    out_path = args.out_candidates
    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    print(f"[retrieve] saved candidates: {out_path} ({len(rows_out)} rows)")

# ========== 3) RERANK (light mix) ==========
def cmd_rerank(args):
    df = pd.read_csv(args.candidates)
    weights = json.loads(args.weights)
    alpha = weights.get("alpha", 0.6)
    beta  = weights.get("beta", 0.15)
    gamma = weights.get("gamma", 0.25)
    eps   = weights.get("eps", 0.05)

    rows = []
    for qid, grp in df.groupby("q_id"):
        bm = grp["bm25"].to_numpy(float)
        d1 = grp["dense_h1"].to_numpy(float)
        d2 = grp["dense_h2"].to_numpy(float)
        rrf = grp["rrf"].to_numpy(float)
        th = grp["title_hit"].to_numpy(float)

        bm_s = minmax_scale(bm)
        dmax = minmax_scale(np.maximum(d1, d2))
        rrf_s = minmax_scale(rrf)

        score = alpha*rrf_s + beta*bm_s + gamma*dmax + eps*th
        tmp = grp.copy()
        tmp["score_chunk"] = score
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.out_chunks, index=False)
    print(f"[rerank] saved: {args.out_chunks}")

# ========== 4) AGGREGATE ==========
def logsumexp(a: np.ndarray, axis=None, keepdims=False):
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True) + 1e-12)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

def cmd_aggregate(args):
    df = pd.read_csv(args.rerank)
    rows = []
    for qid, grp in df.groupby("q_id"):
        doc_scores = {}
        for web_id, g2 in grp.groupby("web_id"):
            g2s = g2.sort_values("score_chunk", ascending=False).head(args.topn)
            sc = g2s["score_chunk"].to_numpy(float)
            if args.mode == "logsumexp3":
                agg = float(logsumexp(sc))
            elif args.mode == "sum3":
                agg = float(np.sum(sc))
            elif args.mode == "max":
                agg = float(np.max(sc))
            else:
                raise ValueError("unknown mode")
            doc_scores[web_id] = agg

        top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        for web_id, agg_score in top_docs:
            rows.append({"q_id": qid, "web_id": web_id, "agg_score": agg_score})

    out = pd.DataFrame(rows)
    out.to_csv(args.out_docs, index=False)
    print(f"[aggregate] saved: {args.out_docs}")

# ========== 5) SUBMIT ==========
def cmd_submit(args):
    import json
    df = pd.read_csv(args.docs)
    out_rows = []
    K = args.k

    for qid, grp in df.groupby("q_id"):
        top_docs = (grp.sort_values("agg_score", ascending=False)
                       .loc[:, "web_id"].astype(str).tolist())

        seen = set()
        uniq = []
        for d in top_docs:
            if d not in seen:
                uniq.append(d); seen.add(d)

        if len(uniq) == 0:
            uniq = ["0"] * K
        while len(uniq) < K:
            uniq.append(uniq[-1])

        def to_int_safe(x):
            try: return int(x)
            except: return x
        web_list = [to_int_safe(x) for x in uniq[:K]]

        out_rows.append({
            "q_id": int(qid),
            "web_list": json.dumps(web_list, ensure_ascii=False)  # "[1, 2, 3, 4, 5]"
        })

    sub = pd.DataFrame(out_rows)
    sub.to_csv(args.out, index=False)
    print(f"[submit] saved: {args.out}")

# ========== CLI ==========
def main():
    p = argparse.ArgumentParser(description="Minimal RAG retrieval core")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="Ingest → Chunk → BM25+Embeddings")
    b.add_argument("--corpus", required=True, help="CSV with columns: web_id,url,title,type,text")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--h1-size", type=int, default=1000)
    b.add_argument("--h1-overlap", type=int, default=200)
    b.add_argument("--h2-size", type=int, default=300)
    b.add_argument("--h2-overlap", type=int, default=60)
    b.add_argument("--k1", type=float, default=1.4)
    b.add_argument("--b", type=float, default=0.35)
    b.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    b.add_argument("--batch", type=int, default=64)
    b.add_argument("--hnsw-M", type=int, default=64)
    b.add_argument("--hnsw-efC", type=int, default=300)
    b.set_defaults(func=cmd_build_index)

    r = sub.add_parser("retrieve", help="BM25+Dense → union → RRF → candidates.csv")
    r.add_argument("--questions", required=True, help="CSV with columns: q_id,query")
    r.add_argument("--index-dir", required=True)
    r.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    r.add_argument("--top-lex", type=int, default=400)
    r.add_argument("--top-dense", type=int, default=400)
    r.add_argument("--rrf-k", type=int, default=60)
    r.add_argument("--pool-size", type=int, default=800)
    r.add_argument("--out-candidates", required=True)
    r.set_defaults(func=cmd_retrieve)

    rr = sub.add_parser("rerank", help="Light mix → score_chunk")
    rr.add_argument("--candidates", required=True)
    rr.add_argument("--weights", default='{"alpha":0.6,"beta":0.15,"gamma":0.25,"eps":0.05}')
    rr.add_argument("--out-chunks", required=True)
    rr.set_defaults(func=cmd_rerank)

    ag = sub.add_parser("aggregate", help="Chunk→Doc aggregation")
    ag.add_argument("--rerank", required=True)
    ag.add_argument("--mode", choices=["logsumexp3","sum3","max"], default="logsumexp3")
    ag.add_argument("--topn", type=int, default=3)
    ag.add_argument("--out-docs", required=True)
    ag.set_defaults(func=cmd_aggregate)

    sb = sub.add_parser("submit", help="Top-K docs → submit.csv")
    sb.add_argument("--docs", required=True)
    sb.add_argument("--k", type=int, default=5)
    sb.add_argument("--out", required=True)
    sb.set_defaults(func=cmd_submit)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
