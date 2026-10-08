"""Pure-python replica of the mmBERT/Gemma tokenizer (tokenizers 0.23 semantics).

Reads vocab / merges / added-tokens straight from tokenizer.json and
re-implements the tokenizers encode pipeline with no third-party runtime:

  1. added-token extraction on the RAW text (longest match at each position);
  2. per plain segment: normalizer (" " -> U+2581), Metaspace pre-tokenization
     (split on U+2581, every piece U+2581-prefixed, prepend_scheme=always);
  3. byte-level BPE over the merges rank table with byte fallback.

Byte order and merge semantics follow the tokenizers Rust implementation
(models/bpe/model.rs, models/bpe/word.rs): lowest-rank adjacent pair is merged
first; ranks are unique so the result is deterministic. Verified against the
reference `tokenizers` build on a differential corpus (dev_tools/verify_tokenizer.py).

decode() is used only for the startup vocabulary sanity check.
"""

import heapq

META = "\u2581"  # metaspace replacement character (U+2581 LOWER ONE EIGHTH BLOCK)


class GemmaTokenizer:
    def __init__(self, tok_dir):
        import json

        with open(tok_dir + "/tokenizer.json", encoding="utf-8") as fh:
            j = json.load(fh)
        m = j["model"]
        self.vocab = m["vocab"]  # token -> id
        self.id2tok = {v: k for k, v in self.vocab.items()}
        # merges: id-pair -> (rank, merged_id); the merged token string is a + b
        self.merges = {}
        for rank, (a, b) in enumerate(m["merges"]):
            self.merges[(self.vocab[a], self.vocab[b])] = (rank, self.vocab[a + b])
        self.unk_token = m["unk_token"]
        self.unk_id = self.vocab[self.unk_token]
        self.pad_id = self.vocab["<pad>"]
        self.eos_id = self.vocab["<eos>"]
        self.bos_id = self.vocab["<bos>"]
        self.mask_id = self.vocab["<mask>"]
        self.mask_token = "<mask>"
        # build_sequence uses the tokenizer-level cls/sep ids (bos/eos here)
        self.cls_token_id = self.bos_id
        self.sep_token_id = self.eos_id
        # added tokens: matched against RAW text, longest first
        self.added = {}  # content -> id
        for a in j["added_tokens"]:
            self.added[a["content"]] = a["id"]
        self._by_char = {}
        for content in sorted(self.added, key=len, reverse=True):
            self._by_char.setdefault(content[0], []).append(content)
        # byte tokens <0xXX> -> id (255 of 256 present; 0x09 is covered by
        # the literal tab char in the vocab)
        self.byte_ids = {}
        for b in range(256):
            key = "<0x%02X>" % b
            if key in self.vocab:
                self.byte_ids[b] = self.vocab[key]
        # per-piece BPE cache (bounded)
        self._cache = {}
        self._cache_limit = 200000

    # ------------------------------------------------------------------ encode
    def encode(self, text, truncation=False, max_length=None):
        """Equivalent of tokenizers encode(text, add_special_tokens=False)."""
        if not text:
            return []
        ids = []
        plain = []
        i, n = 0, len(text)
        while i < n:
            cand = self._by_char.get(text[i])
            hit = None
            if cand is not None:
                for content in cand:
                    if text.startswith(content, i):
                        hit = content
                        break
            if hit is None:
                plain.append(text[i])
                i += 1
            else:
                if plain:
                    ids.extend(self._encode_plain("".join(plain)))
                    plain = []
                ids.append(self.added[hit])
                i += len(hit)
        if plain:
            ids.extend(self._encode_plain("".join(plain)))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

    def _encode_plain(self, s):
        # normalizer: U+0020 -> U+2581
        s = s.replace(" ", META)
        if not s:
            return []
        # metaspace: split on U+2581; every piece gets a U+2581 prefix
        parts = s.split(META)
        if parts[0] == "":
            pieces = [META + p for p in parts[1:]]
        else:
            pieces = [META + parts[0]]
            pieces.extend(META + p for p in parts[1:])
        out = []
        for piece in pieces:
            out.extend(self._bpe(piece))
        return out

    def _bpe(self, piece):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        syms = []
        for ch in piece:
            cid = self.vocab.get(ch)
            if cid is not None:
                syms.append(cid)
            else:
                for b in ch.encode("utf-8"):
                    syms.append(self.byte_ids.get(b, self.unk_id))
        n = len(syms)
        if n <= 1:
            result = syms
        elif n <= 32:
            result = self._bpe_scan(syms)
        else:
            result = self._bpe_heap(syms)
        if len(self._cache) < self._cache_limit:
            self._cache[piece] = result
        return result

    def _bpe_scan(self, syms):
        """Naive O(n^2) merge loop: fine for typical short pieces."""
        merges = self.merges
        while True:
            best_rank = None
            best_pos = -1
            best_new = -1
            last = len(syms) - 1
            for j in range(last):
                m = merges.get((syms[j], syms[j + 1]))
                if m is not None and (best_rank is None or m[0] < best_rank):
                    best_rank, best_new, best_pos = m[0], m[1], j
            if best_pos < 0:
                break
            syms[best_pos:best_pos + 2] = [best_new]
        return syms

    def _bpe_heap(self, syms):
        """Priority-queue merge loop mirroring the Rust merge_all."""
        merges = self.merges
        heap = []
        for j in range(len(syms) - 1):
            m = merges.get((syms[j], syms[j + 1]))
            if m is not None:
                heap.append((m[0], j, m[1]))
        heapq.heapify(heap)
        n = len(syms)
        alive = [True] * n
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n))
        nxt.append(-1)
        new_ids = syms[:]
        while heap:
            rank, pos, new_id = heapq.heappop(heap)
            if not alive[pos] or nxt[pos] == -1:
                continue
            right = nxt[pos]
            m = merges.get((new_ids[pos], new_ids[right]))
            if m is None or m[0] != rank:
                continue
            new_ids[pos] = new_id
            alive[right] = False
            rn = nxt[right]
            nxt[pos] = rn
            if rn != -1:
                prev[rn] = pos
            if prev[pos] != -1:
                m2 = merges.get((new_ids[prev[pos]], new_id))
                if m2 is not None:
                    heapq.heappush(heap, (m2[0], prev[pos], m2[1]))
            if nxt[pos] != -1:
                m3 = merges.get((new_id, new_ids[nxt[pos]]))
                if m3 is not None:
                    heapq.heappush(heap, (m3[0], pos, m3[1]))
        return [new_ids[j] for j in range(n) if alive[j]]

    # ------------------------------------------------------------------ decode
    def decode(self, ids):
        """Join token strings back into text (bytes via <0xXX>, U+2581 -> space).

        Only used for the startup vocabulary sanity check; the model serving
        path never decodes.
        """
        pieces = []
        for i in ids:
            t = self.id2tok.get(i)
            if t is None:
                continue
            if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                pieces.append(bytes([int(t[3:5], 16)]))
            else:
                pieces.append(t)
        out = []
        buf = bytearray()

        def flush():
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf.clear()

        for p in pieces:
            if isinstance(p, bytes):
                buf.extend(p)
            else:
                flush()
                out.append(p)
        flush()
        return "".join(out).replace(META, " ")
