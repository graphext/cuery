from __future__ import annotations

import os
import re
import random
import unicodedata
from collections import defaultdict
from urllib.parse import urlparse
from typing import Any

import httpx
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


async def expand_competitors(brands: list[dict], *, sector: str, market: str, max_count: int, notes: list[str]) -> list[dict]:
    if not brands:
        return brands
    seed = brands[0]["brand"]
    guesses = [f"{seed} COMP{i}" for i in range(1, 6)][: max_count]
    comps = [
        {
            "brand": g,
            "aliases": [g, g.lower()],
            "domains": [f"{g.split()[0].lower()}.com"],
        }
        for g in guesses
    ]
    merged = {b["brand"]: b for b in [*brands, *comps]}
    notes.append(f"Competitors guessed for demo: {[c['brand'] for c in comps]}")
    return list(merged.values())


# ----------------------------- Helpers: prompts -----------------------------

async def generate_commercial_prompts(*, n: int, language: str, sector: str, market: str, focus_brands: list[str], random_seed: int) -> list[str]:
    random.seed(int(random_seed))
    yrs = ["2023", "2024", "2025"]
    brand = focus_brands[0] if focus_brands else "brand"
    templates = [
        f"best {sector} providers in {market} {random.choice(yrs)}",
        f"alternatives to {brand.lower()} in {market}",
        f"{sector} prices and coverage {market}",
        f"reviews and complaints about {sector} companies in {market}",
        f"how to cancel {sector} policy and switch in {market}",
        f"top {sector} for freelancers in {market} {random.choice(yrs)}",
        f"ranking of {sector} companies in {market}",
        f"{sector} for families in {market}",
        f"{sector} for students in {market}",
        f"{sector} with international coverage {market}",
    ]
    # Greedy sampling with simple near-duplicate filtering (RapidFuzz if available)
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

    while len(bag) < int(n):
        q = random.choice(templates)
        if not is_dupe(q, bag):
            bag.append(q)
    return bag


# ------------------------------ Helpers: LLMs -------------------------------

async def run_llms(*, prompts: list[str], llms: list[str], repetitions: int, timeouts: dict) -> list[dict]:
    out: list[dict[str, Any]] = []
    timeout_sec = int((timeouts or {}).get("llm_call", 45))
    for prompt in prompts:
        for llm in llms:
            for r in range(1, int(repetitions) + 1):
                text = await _call_llm(llm, prompt, timeout_sec)
                out.append({
                    "prompt": prompt,
                    "llm": llm,
                    "repetition": r,
                    "response_text": text or "",
                })
    return out


async def _call_llm(llm_id: str, prompt: str, timeout_sec: int) -> str:
    if llm_id.startswith("openai:"):
        key = os.environ["OPENAI_API_KEY"]
        model = llm_id.split(":", 1)[1]
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            )
            j = r.json()
            return j.get("choices", [{}])[0].get("message", {}).get("content", "")
    if llm_id.startswith("google:"):
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        model = llm_id.split(":", 1)[1]
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
            j = r.json()
            parts = j.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            return " ".join([p.get("text", "") for p in parts])
    if llm_id.startswith("perplexity:"):
        key = os.environ["PERPLEXITY_API_KEY"]
        model = llm_id.split(":", 1)[1]
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            r = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            )
            j = r.json()
            return j.get("choices", [{}])[0].get("message", {}).get("content", "")
    raise ValueError(f"Unsupported LLM id: {llm_id}")


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
        actor_input = await Actor.get_input() or {}
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
        brands = await merge_brand_overrides(base_brands, brand_overrides)

        if expand:
            brands = await expand_competitors(
                brands, sector=sector, market=market, max_count=max_competitors, notes=notes
            )

        prompts = await generate_commercial_prompts(
            n=n_prompts,
            language=language,
            sector=sector,
            market=market,
            focus_brands=[b["brand"] for b in brands],
            random_seed=random_seed,
        )

        runs = await run_llms(
            prompts=prompts, llms=llms, repetitions=repetitions, timeouts=timeouts_sec
        )

        result = await _aggregate_mentions_matrix(runs=runs, brands=brands)

        # Persist artifacts (Apify KV store + Dataset)
        await Actor.set_value("brands.json", brands)
        await Actor.set_value("prompts.json", prompts)
        await Actor.set_value("runs.json", runs)
        for row in result["matrix"]:
            await Actor.push_data(row)
        await Actor.set_value("stats.json", result["stats"])
        await Actor.set_value("notes.json", notes)
        await Actor.set_value(
            "OUTPUT",
            {
                "brands": brands,
                "promptsCount": len(prompts),
                "stats": result["stats"],
                "notes": notes[:10],
            },
        )


