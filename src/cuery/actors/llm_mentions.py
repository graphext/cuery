from __future__ import annotations

import os
import asyncio
import re
import random
import unicodedata
from collections import defaultdict
from urllib.parse import urlparse
from typing import Any

import httpx
import time
from datetime import datetime, timezone
from apify import Actor


SYSTEM_PROMPT = (
    "You receive: (1) company URLs, (2) expand_competitors flag, (3) n_prompts,\n"
    "(4) target LLMs with repetitions. Normalize brands from URLs, optionally expand\n"
    "competitors, generate n_prompts of realistic commercial intent for the specified\n"
    "market and language, execute each prompt repetitions times per LLM, detect\n"
    "mentions by brand name and by URL, and produce a matrix (rows=prompts; columns=LLM x brand\n"
    "with by_name/by_url booleans). Deduplicate prompts semantically, cover intents\n"
    "(comparatives, transactional, alternatives, trust, regulation/location), localize,\n"
    "and log issues in notes. Be deterministic if random_seed provided."
)


# ----------------------------- Helpers: timing -------------------------------

def _format_ms(ms: int) -> str:
    seconds, ms_rem = divmod(int(ms), 1000)
    minutes, secs = divmod(seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins}m {secs}.{ms_rem:03d}s"
    if minutes:
        return f"{mins}m {secs}.{ms_rem:03d}s"
    if seconds:
        return f"{secs}.{ms_rem:03d}s"
    return f"{ms} ms"

def _iso(ts_unix: float) -> str:
    """Return ISO 8601 UTC timestamp to seconds precision."""
    try:
        return datetime.fromtimestamp(ts_unix, tz=timezone.utc).replace(microsecond=0).isoformat()
    except Exception:
        return ""


def _dedupe(urls: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _extract_http_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    try:
        matches = re.findall(r"https?://[\w\-\./%#?=&]+", text)
        return _dedupe(matches)
    except Exception:
        return []


def _collect_urls_from_json(obj: Any) -> list[str]:
    urls: list[str] = []
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (str, bytes)):
                    if isinstance(v, bytes):
                        try:
                            v = v.decode("utf-8", errors="ignore")  # type: ignore
                        except Exception:
                            continue
                    if k.lower() in {"url", "uri", "source_url"} and str(v).startswith(("http://", "https://")):
                        urls.append(str(v))
                else:
                    urls.extend(_collect_urls_from_json(v))
        elif isinstance(obj, list):
            for it in obj:
                urls.extend(_collect_urls_from_json(it))
    except Exception:
        return []
    return _dedupe(urls)

# ----------------------------- Helpers: brands ------------------------------

def _norm_str(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


async def normalize_brands_from_urls(urls: list[str], notes: list[str]) -> list[dict]:
    out: list[dict[str, Any]] = []
    for u in urls or []:
        try:
            host = urlparse(u).netloc or ""
            root = host.lower().removeprefix("www.")
            name_guess = root.split(".")[0] if root else ""
            brand = name_guess.upper() if name_guess else (host or u)
            domains = [d for d in {root, host} if d]
            out.append({
                "brand": brand,
                "aliases": [brand, brand.lower()],
                "domains": domains,
            })
        except Exception as e:  # pragma: no cover - defensive
            notes.append(f"Failed to normalize brand from URL {u}: {e}")
    uniq = {b["brand"]: b for b in out}
    return list(uniq.values())


async def merge_brand_overrides(brands: list[dict], overrides: list[dict]) -> list[dict]:
    by = {
        b["brand"]: {
            "brand": b["brand"],
            "aliases": list(b.get("aliases", [])),
            "domains": list(b.get("domains", [])),
        }
        for b in brands
    }
    for o in overrides or []:
        curr = by.get(o["brand"], {"brand": o["brand"], "aliases": [], "domains": []})
        curr["aliases"] = sorted(set(curr["aliases"] + list(o.get("aliases", []))))
        curr["domains"] = sorted(set(curr["domains"] + list(o.get("domains", []))))
        by[o["brand"]] = curr
    return list(by.values())


async def expand_competitors(
    brands: list[dict],
    *,
    sector: str,
    market: str,
    max_count: int,
    notes: list[str],
    language: str | None = None,
    urls: list[str] | None = None,
    timeouts: dict | None = None,
) -> list[dict]:
    """Discover competitors using GPT-5 via OpenAI Responses (no fallback).

    Returns original brands plus discovered competitors (up to max_count).
    Does not synthesize placeholders when the model fails.
    """
    # Build instruction for LLM
    seed_brands = ", ".join([b.get("brand", "").strip() for b in brands if b.get("brand")])
    url_list = ", ".join((urls or [])[:10]) if urls else ""
    instruction = (
        f"Identify up to {max_count} real competitor brands for: [{seed_brands}] in sector '{sector}' within market '{market}'. "
        f"Return a strict JSON array of objects with fields: brand (string), aliases (string[]), domains (string[]). "
        f"Use each company's primary website domains (no social/aggregators). "
        f"Only output the JSON array, nothing else. Context URLs (optional): [{url_list}]."
    )

    timeout_sec = int((timeouts or {}).get("llm_call", 45))
    llm_id = "openai:gpt-5"
    text, meta = await _call_llm(llm_id, instruction, timeout_sec, language)

    # Parse JSON array of objects robustly
    import json
    import re as _re

    def _extract_json_array_objs(s: str) -> list[dict]:
        if not s:
            return []
        s2 = s.strip()
        if s2.startswith("```") and s2.endswith("```"):
            lines = s2.splitlines()
            s2 = "\n".join(lines[1:-1])
        m = _re.search(r"\[.*\]", s2, flags=_re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                return arr if isinstance(arr, list) else []
            except Exception:
                pass
        try:
            arr = json.loads(s2)
            return arr if isinstance(arr, list) else []
        except Exception:
            return []

    raw_items = _extract_json_array_objs(text)
    comps: list[dict] = []
    seen_names: set[str] = set()
    for it in raw_items:
        try:
            brand_name = str(it.get("brand", "")).strip()
            if not brand_name:
                continue
            key = brand_name.upper()
            if key in seen_names:
                continue
            seen_names.add(key)
            aliases = [brand_name, brand_name.lower(), *[str(a).strip() for a in (it.get("aliases") or []) if a]]
            # Normalize domains, keep host-like strings, dedupe www/root
            doms_in = [str(d).strip().lower() for d in (it.get("domains") or []) if d]
            doms_norm: list[str] = []
            for d in doms_in:
                dd = d.removeprefix("www.")
                if dd and dd not in doms_norm:
                    doms_norm.append(dd)
                w = f"www.{dd}" if dd else ""
                if dd and w not in doms_norm:
                    # add www variant for URL regex detection later
                    doms_norm.append(w)
            comps.append({
                "brand": key,
                "aliases": sorted(set(a for a in aliases if a)),
                "domains": [d for d in doms_norm if d],
            })
            if len(comps) >= max_count:
                break
        except Exception:
            continue

    if not comps:
        # Do not fabricate placeholders; keep original brands
        status = meta.get("status")
        err = meta.get("error")
        if err or status and status != 200:
            notes.append(f"Competitor discovery via GPT-5 returned no results (status={status} error={err})")
        else:
            notes.append("Competitor discovery via GPT-5 returned no parseable results")
        return brands

    merged = {b["brand"]: b for b in brands}
    for c in comps:
        if c["brand"] not in merged:
            merged[c["brand"]] = c
    notes.append(f"Competitors discovered via GPT-5: {[c['brand'] for c in comps]}")
    return list(merged.values())


# ----------------------------- Helpers: prompts -----------------------------

async def generate_commercial_prompts(
    *,
    n: int,
    language: str,
    sector: str,
    market: str,
    focus_brands: list[str],
    urls: list[str],
    llms: list[str],
    timeouts: dict,
    random_seed: int,
) -> list[str]:
    """Generate N realistic commercial/consumer search queries using an LLM meta-instruction.

    Avoids hardcoded templates. Leverages the first configured LLM (or a default) to synthesize
    localized prompts, returning a JSON array of strings. Includes deduplication.
    """
    random.seed(int(random_seed))
    meta_llm = llms[0] if llms else "openai:gpt-4.1-mini"
    lang = language or "en"
    brand_list = ", ".join(focus_brands[:10]) if focus_brands else ""
    url_list = ", ".join(urls[:10]) if urls else ""
    instruction = (
        f"Generate {n} unique, concise search queries in {lang} for consumer/commercial intent in the sector '{sector}' "
        f"in market '{market}'. Cover realistic user intents like comparisons, transactional queries, alternatives, trust/regulatory, and location nuances. "
        f"If brand context helps, consider these brands: [{brand_list}]. If helpful, infer from URLs: [{url_list}]. "
        f"Strictly return a JSON array of strings. No numbering, no prose, no code fences."
    )

    timeout_sec = int((timeouts or {}).get("llm_call", 45))
    text, _ = await _call_llm(meta_llm, instruction, timeout_sec, language)

    # Parse JSON array robustly
    import json
    import re as _re

    def _extract_json_array(s: str) -> list[str]:
        if not s:
            return []
        s2 = s.strip()
        # Remove code fences if present
        if s2.startswith("```") and s2.endswith("```"):
            lines = s2.splitlines()
            s2 = "\n".join(lines[1:-1])
        # Try to find first JSON array
        m = _re.search(r"\[.*\]", s2, flags=_re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if isinstance(x, (str, int, float)) or isinstance(x, str)]
            except Exception:
                pass
        # Try entire string
        try:
            arr = json.loads(s2)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if isinstance(x, (str, int, float)) or isinstance(x, str)]
        except Exception:
            return []
        return []

    candidates = _extract_json_array(text)

    # Greedy deduplication
    bag: list[str] = []

    def similar(a: str, b: str) -> bool:
        try:
            from rapidfuzz import fuzz  # type: ignore
            return fuzz.token_set_ratio(a, b) >= 90
        except Exception:
            return _norm_str(a) == _norm_str(b)

    def is_dupe(q: str, acc: list[str]) -> bool:
        for x in acc:
            if similar(q, x):
                return True
        return False

    for q in candidates:
        if len(bag) >= int(n):
            break
        q = (q or "").strip()
        if q and not is_dupe(q, bag):
            bag.append(q)

    # If LLM returned fewer than requested, clone mild variants
    i = 0
    while len(bag) < int(n) and candidates:
        base = (candidates[i % len(candidates)] or "").strip()
        variant = base
        if variant and not is_dupe(variant, bag):
            bag.append(variant)
        i += 1
    return bag


# ------------------------------ Helpers: LLMs -------------------------------

async def run_llms(*, prompts: list[str], llms: list[str], repetitions: int, timeouts: dict, max_concurrent: int, language: str) -> list[dict]:
    out: list[dict[str, Any]] = []
    timeout_sec = int((timeouts or {}).get("llm_call", 45))
    jobs: list[tuple[str, str, int]] = []
    for prompt in prompts:
        for llm in llms:
            for r in range(1, int(repetitions) + 1):
                jobs.append((prompt, llm, r))

    total = len(jobs)
    step = max(1, total // 20)
    Actor.log.info(
        f"Running LLMs concurrently: prompts={len(prompts)} llms={len(llms)} repetitions={repetitions} total_calls={total} max_concurrent={max_concurrent}"
    )

    # Open a secondary dataset to stream per-call progress, unique per run to avoid collisions
    ds_name = f"runs-{int(time.time())}"
    runs_ds = await Actor.open_dataset(name=ds_name)

    sem = asyncio.Semaphore(max(1, int(max_concurrent)))
    idx = 0

    async def worker(job: tuple[str, str, int]) -> dict:
        prompt, llm, rep = job
        started_at_unix = time.time()
        started_perf = time.perf_counter()
        async with sem:
            text, meta = await _call_llm(llm, prompt, timeout_sec, language)
        finished_perf = time.perf_counter()
        finished_at_unix = time.time()
        record = {
            "prompt": prompt,
            "llm": llm,
            "repetition": rep,
            "response_text": text or "",
            "status": meta.get("status"),
            "error": meta.get("error"),
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "response_preview": (text or "")[:200],
            "grounding_urls": meta.get("grounding_urls", []),
            "sources_urls": meta.get("sources_urls", []),
            "language": language,
            "started_at": _iso(started_at_unix),
            "finished_at": _iso(finished_at_unix),
            "duration_s": round((finished_perf - started_perf), 3),
        }
        try:
            await runs_ds.push_data(record)
        except Exception:
            pass
        return record

    tasks = [asyncio.create_task(worker(j)) for j in jobs]
    for fut in asyncio.as_completed(tasks):
        rec = await fut
        out.append(rec)
        idx += 1
        if idx % step == 0 or idx == total:
            Actor.log.info(f"LLM calls progress: {idx}/{total} ({int(idx/total*100)}%)")
    return out


def _parse_llm_id(llm_id: str) -> tuple[str, str]:
    if ":" in llm_id:
        provider, model = llm_id.split(":", 1)
        return provider.lower(), model
    if "/" in llm_id:
        provider, model = llm_id.split("/", 1)
        return provider.lower(), model
    raise ValueError(f"Unsupported LLM id format: {llm_id}")


def _supports_openai_grounding(model: str) -> bool:
    m = model.lower()
    # Responses web_search generally available on 4.1/5 families, including mini.
    # We attempt Responses even for chat-labeled variants (we'll map to a base model name).
    return ("gpt-4.1" in m) or ("gpt-5" in m)


def _supports_gemini_grounding(model: str) -> bool:
    m = model.lower()
    # googleSearchRetrieval is supported on 1.5/2.5 families, but not all endpoints/tiers accept it.
    # If API returns 400 INVALID_ARGUMENT, we will retry without tools.
    return ("gemini-2.5" in m) or ("gemini-1.5" in m)


async def _call_llm(llm_id: str, prompt: str, timeout_sec: int, language: str | None = None) -> tuple[str, dict]:
    provider, model = _parse_llm_id(llm_id)
    if provider == "openai":
        key = os.environ["OPENAI_API_KEY"]
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            # Try Responses with web_search when supported
            if _supports_openai_grounding(model):
                try:
                    r = await client.post(
                        "https://api.openai.com/v1/responses",
                        headers={"Authorization": f"Bearer {key}"},
                        json={
                            "model": model,
                            "input": prompt,
                            "tools": [{"type": "web_search"}],
                        },
                    )
                    if r.status_code == 200:
                        j = r.json()
                        sources: list[str] = []
                        # references array
                        for ref in j.get("references", []) or []:
                            u = ref.get("url") or ref.get("uri")
                            if u:
                                sources.append(u)
                        # output messages annotations
                        for item in j.get("output", []) or []:
                            if item.get("type") == "message":
                                for c in item.get("content", []) or []:
                                    for ann in c.get("annotations", []) or []:
                                        u = ann.get("url") or ann.get("file_path")
                                        if u:
                                            sources.append(u)
                        # also parse inline links from text if any
                        inline = _extract_http_urls_from_text(j.get("output_text", ""))
                        sources.extend(inline)
                        # Deep-scan JSON for any url/uri keys
                        sources.extend(_collect_urls_from_json(j))
                        if isinstance(j, dict) and j.get("output_text"):
                            return j["output_text"], {"status": r.status_code, "provider": provider, "model": model, "grounding_urls": [], "sources_urls": _dedupe(sources)}
                        texts: list[str] = []
                        # reuse sources list for any additional links parsed from text
                        for item in j.get("output", []) or []:
                            if item.get("type") == "message":
                                for c in item.get("content", []) or []:
                                    t = c.get("text") or c.get("content") or ""
                                    if t:
                                        texts.append(t)
                                    for ann in c.get("annotations", []) or []:
                                        u = ann.get("url") or ann.get("file_path")
                                        if u:
                                            sources.append(u)
                                    sources.extend(_extract_http_urls_from_text(t))
                        if texts:
                            # Also try to collect URLs from the full JSON response, just in case
                            sources.extend(_collect_urls_from_json(j))
                            return "\n".join(texts), {"status": r.status_code, "provider": provider, "model": model, "grounding_urls": [], "sources_urls": _dedupe(sources)}
                except Exception as e:
                    # No fallback behavior: return error
                    return "", {"status": None, "error": f"responses_call_failed: {e}", "provider": provider, "model": model}

            # No fallback behavior requested: do not call Chat Completions
            return "", {"status": None, "error": "no_fallback_chat_completions", "provider": provider, "model": model}
    if provider == "google":
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            async def call_gemini(with_grounding: bool) -> tuple[str, dict]:
                payload: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
                if with_grounding and _supports_gemini_grounding(model):
                    m = model.lower()
                    # Per Google docs: gemini-2.x uses google_search; 1.5 uses google_search_retrieval
                    if "gemini-2.5" in m or "gemini-2." in m:
                        payload["tools"] = [{"google_search": {}}]
                    else:
                        payload["tools"] = [{"google_search_retrieval": {}}]
                payload["generationConfig"] = {"temperature": 0.3}
                if language:
                    payload["safetySettings"] = []
                    payload["systemInstruction"] = {"parts": [{"text": f"Responde en {language} con baja verbosidad."}]}
                try:
                    rr = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                        json=payload,
                    )
                except Exception as e:
                    return "", {"status": None, "error": f"gemini request error: {e}", "provider": provider, "model": model}
                if rr.status_code != 200:
                    return "", {"status": rr.status_code, "error": rr.text, "provider": provider, "model": model}
                jj = rr.json()
                texts: list[str] = []
                sources: list[str] = []
                goog_urls: list[str] = []
                for cand in jj.get("candidates", []) or []:
                    parts = cand.get("content", {}).get("parts", [])
                    for p in parts:
                        t = p.get("text", "")
                        if t:
                            texts.append(t)
                        # inline sources if any
                        sources.extend(_extract_http_urls_from_text(t))
                    gm = cand.get("groundingMetadata", {}) or {}
                    # 1) Primary: groundingAttributions -> sourceId.web.uri
                    for ga in gm.get("groundingAttributions", []) or []:
                        src = (((ga.get("sourceId") or {}).get("web") or {}).get("uri"))
                        if src:
                            sources.append(src)
                        # Sometimes appears directly under .web.uri
                        src2 = ((ga.get("web") or {}).get("uri"))
                        if src2:
                            sources.append(src2)
                    # 2) Alternative: citations list
                    for cit in cand.get("citationMetadata", {}).get("citations", []) or []:
                        u = cit.get("uri") or cit.get("url")
                        if u:
                            sources.append(u)
                    # 3) Grounding entrypoint (Google/Vertex links)
                    sep = gm.get("searchEntryPoint") or {}
                    for val in (sep.values() if isinstance(sep, dict) else []):
                        if isinstance(val, str):
                            for m in re.findall(r"https?://[\w\-\./%#?=&]+", val):
                                if "vertexai" in m or "vertexsearch" in m or "google.com" in m:
                                    goog_urls.append(m)
                # Resolve Vertex redirectors (without making network calls) by decoding 'q' param when present
                def resolve_vertex(u: str) -> str:
                    try:
                        from urllib.parse import urlparse, parse_qs
                        import base64
                        import re as _re2
                        pu = urlparse(u)
                        host = pu.netloc
                        if (host.endswith("vertexsearch.cloud.google.com") or host.endswith("vertexaisearch.cloud.google.com")) and (
                            pu.path.startswith("/grounding-api-redirect/") or pu.path.startswith("/grounding-api-redirect")
                        ):
                            q = parse_qs(pu.query).get("q")
                            if q and isinstance(q, list) and q[0]:
                                return q[0]
                            # Try to decode last path segment as base64-url and extract first URL inside
                            seg = pu.path.split("/")[-1]
                            if seg:
                                pad = "=" * ((4 - len(seg) % 4) % 4)
                                try:
                                    decoded = base64.urlsafe_b64decode((seg + pad).encode("utf-8"))
                                    text = decoded.decode("utf-8", errors="ignore")
                                    m = _re2.search(r"https?://[\w\-\./%#?=&]+", text)
                                    if m:
                                        return m.group(0)
                                except Exception:
                                    pass
                        return u
                    except Exception:
                        return u

                # Keep Google grounding URLs as-is (vertex/google links), do not replace with source pages
                # Also attempt to decode vertex redirector to a canonical Google URL when possible
                resolved = [resolve_vertex(u) for u in goog_urls]

                # Attempt HTTP resolution for Vertex redirectors that could not be decoded
                async def _resolve_http(u: str) -> str:
                    try:
                        if not u:
                            return u
                        from urllib.parse import urlparse as _up
                        pu = _up(u)
                        if (pu.netloc.endswith("vertexsearch.cloud.google.com") or pu.netloc.endswith("vertexaisearch.cloud.google.com")) and (
                            pu.path.startswith("/grounding-api-redirect/") or pu.path.startswith("/grounding-api-redirect")
                        ):
                            r0 = await client.get(u, follow_redirects=False, timeout=timeout_sec)
                            loc = r0.headers.get("location") or r0.headers.get("Location")
                            if loc:
                                return loc
                            r1 = await client.get(u, follow_redirects=True, timeout=timeout_sec)
                            return str(r1.url)
                        return u
                    except Exception:
                        return u

                # Only attempt network resolution for those still on vertex domain
                unresolved_idxs = []
                for i, ru in enumerate(resolved):
                    try:
                        from urllib.parse import urlparse as _up
                        pu = _up(ru)
                        if pu.netloc.endswith("vertexsearch.cloud.google.com") or pu.netloc.endswith("vertexaisearch.cloud.google.com"):
                            unresolved_idxs.append(i)
                    except Exception:
                        pass
                if unresolved_idxs:
                    tasks = [_resolve_http(resolved[i]) for i in unresolved_idxs[:10]]  # cap to 10 to avoid long delays
                    http_resolved = await asyncio.gather(*tasks)
                    for j, idx_i in enumerate(unresolved_idxs[:10]):
                        resolved[idx_i] = http_resolved[j] or resolved[idx_i]

                # Filter only http(s) and drop known placeholders (e.g., SVG namespace)
                filtered: list[str] = []
                for ru in resolved:
                    if not (ru.startswith("http://") or ru.startswith("https://")):
                        continue
                    try:
                        from urllib.parse import urlparse as _up
                        ph = _up(ru)
                        if ph.netloc == "www.w3.org" and ph.path.startswith("/2000/svg"):
                            continue
                    except Exception:
                        pass
                    filtered.append(ru)
                return "\n".join(texts), {"status": rr.status_code, "provider": provider, "model": model, "grounding_urls": _dedupe(filtered), "sources_urls": _dedupe([u for u in sources if u.startswith("http://") or u.startswith("https://")])}

            # Try with grounding; if unsupported, retry without tools
            text, meta = await call_gemini(with_grounding=True)
            if meta.get("status") == 400 and ("INVALID_ARGUMENT" in (meta.get("error") or "") or "not supported" in (meta.get("error") or "").lower()):
                text2, meta2 = await call_gemini(with_grounding=False)
                if text2:
                    return text2, meta2
            return text, meta
    if provider == "perplexity":
        key = os.environ["PERPLEXITY_API_KEY"]
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            r = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            )
            if r.status_code != 200:
                return "", {"status": r.status_code, "error": r.text, "provider": provider, "model": model}
            j = r.json()
            return j.get("choices", [{}])[0].get("message", {}).get("content", ""), {"status": r.status_code, "provider": provider, "model": model}
    return "", {"status": None, "error": f"Unsupported LLM id: {llm_id}", "provider": provider, "model": model}


# --------------------------- Helpers: mentions/matrix ------------------------

def _detect_mentions(text: str, brands: list[dict]) -> list[dict]:
    norm_text = _norm_str(text or "")
    out: list[dict[str, Any]] = []
    for b in brands:
        names = [b["brand"], *b.get("aliases", [])]
        name_hits = [
            n
            for n in names
            if re.search(rf"(^|\W){re.escape(_norm_str(n))}(\W|$)", norm_text)
        ]
        url_hits: list[str] = []
        for d in b.get("domains", []):
            host = re.escape(d.replace("www.", ""))
            m = re.findall(rf"(https?://)?(www\.)?{host}([/\w\-\.\?\=&%#]*)?", text or "", flags=re.I)
            if m:
                url_hits.extend(["".join(x) for x in m])
        out.append({
            "brand": b["brand"],
            "by_name": bool(name_hits),
            "by_url": bool(url_hits),
            "hits": {"names": list(set(name_hits)), "urls": list(set(url_hits))},
        })
    return out


async def _aggregate_mentions_matrix(*, runs: list[dict], brands: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in runs:
        groups[(r["prompt"], r["llm"])].append(r.get("response_text", ""))

    prompts = list({r["prompt"] for r in runs})
    matrix: list[dict[str, Any]] = []
    for p in prompts:
        row: dict[str, Any] = {"prompt": p}
        llms = {r["llm"] for r in runs if r["prompt"] == p}
        for llm in llms:
            reps = groups[(p, llm)]
            merged = {b["brand"]: {"by_name": False, "by_url": False} for b in brands}
            for txt in reps:
                for hit in _detect_mentions(txt, brands):
                    merged[hit["brand"]]["by_name"] |= hit["by_name"]
                    merged[hit["brand"]]["by_url"] |= hit["by_url"]
            for brand, flags in merged.items():
                row[f"{llm}.{brand}.by_name"] = flags["by_name"]
                row[f"{llm}.{brand}.by_url"] = flags["by_url"]
        matrix.append(row)

    def ratio(suffix: str) -> float:
        vals: list[int] = []
        for r in matrix:
            for k, v in r.items():
                if k.endswith(suffix):
                    vals.append(1 if v else 0)
        return (sum(vals) / len(vals)) if vals else 0.0

    stats = {
        "coverage_by_brand": {
            b["brand"]: {
                "by_name": ratio(f"{b['brand']}.by_name"),
                "by_url": ratio(f"{b['brand']}.by_url"),
            }
            for b in brands
        }
    }
    return {"matrix": matrix, "stats": stats}


# --------------------------------- Main -------------------------------------

async def main() -> None:
    async with Actor:
        overall_started = time.perf_counter()
        actor_input = await Actor.get_input() or {}
        Actor.log.info("Starting LLM Mentions Auditor run")
        notes: list[str] = [SYSTEM_PROMPT]

        urls = actor_input.get("urls", [])
        expand = bool(actor_input.get("expand_competitors", True))
        sector = actor_input.get("sector", "unknown")
        market = actor_input.get("market", "Global")
        language = actor_input.get("language", "en-US")
        n_prompts = int(actor_input.get("n_prompts", 30))
        llms = actor_input.get("llms", [])
        repetitions = int(actor_input.get("repetitions", 3))
        brand_overrides = actor_input.get("brand_overrides", [])
        max_competitors = int(actor_input.get("max_competitors", 12))
        random_seed = int(actor_input.get("random_seed", 42))
        timeouts_sec = actor_input.get("timeouts_sec", {"llm_call": 45, "search": 30})

        base_brands = await normalize_brands_from_urls(urls, notes)
        Actor.log.info(f"Normalized {len(base_brands)} brands from URLs")
        brands = await merge_brand_overrides(base_brands, brand_overrides)
        Actor.log.info(f"After overrides: {len(brands)} brands")

        if expand:
            brands = await expand_competitors(
                brands,
                sector=sector,
                market=market,
                max_count=max_competitors,
                notes=notes,
                language=language,
                urls=urls,
                timeouts=timeouts_sec,
            )
            Actor.log.info(f"After competitor expansion: {len(brands)} brands")

        prompts = await generate_commercial_prompts(
            n=n_prompts,
            language=language,
            sector=sector,
            market=market,
            focus_brands=[b["brand"] for b in brands],
            urls=urls,
            llms=llms,
            timeouts=timeouts_sec,
            random_seed=random_seed,
        )
        Actor.log.info(f"Generated {len(prompts)} prompts")

        runs = await run_llms(
            prompts=prompts,
            llms=llms,
            repetitions=repetitions,
            timeouts=timeouts_sec,
            max_concurrent=int(actor_input.get("max_concurrent", 16)),
            language=language,
        )
        Actor.log.info(f"Collected {len(runs)} responses from LLMs")

        result = await _aggregate_mentions_matrix(runs=runs, brands=brands)
        overall_ms = int((time.perf_counter() - overall_started) * 1000)
        Actor.log.info(f"Total runtime: {overall_ms} ms (~{_format_ms(overall_ms)})")
        Actor.log.info(f"Aggregated matrix rows: {len(result.get('matrix', []))}")

        # Persist artifacts (Apify KV store + Dataset)
        await Actor.set_value("brands.json", brands)
        await Actor.set_value("prompts.json", prompts)
        await Actor.set_value("runs.json", runs)
        for row in result["matrix"]:
            await Actor.push_data(row)
        # Add timing stats
        result_stats = dict(result["stats"])
        result_stats["timing_ms"] = {
            "total": overall_ms,
            "per_call_p50": int(sorted(r.get("duration_ms", 0) for r in runs)[len(runs)//2]) if runs else 0,
            "per_call_p95": int(sorted(r.get("duration_ms", 0) for r in runs)[max(0, int(len(runs)*0.95)-1)]) if runs else 0,
        }
        result_stats.setdefault("timing_human", {})
        result_stats["timing_human"]["total"] = _format_ms(overall_ms)
        await Actor.set_value("stats.json", result_stats)
        await Actor.set_value("notes.json", notes)
        await Actor.set_value(
            "OUTPUT",
            {
                "brands": brands,
                "promptsCount": len(prompts),
                "stats": result_stats,
                "notes": notes[:10],
            },
        )


if __name__ == "__main__":
    asyncio.run(main())


