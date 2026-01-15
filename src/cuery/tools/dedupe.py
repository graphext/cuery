"""Tool for semantically de-duplicating entities using LLM-based clustering.

This module provides tools for grouping semantically equivalent entities (phrases, categories,
aspect terms, etc.) into clusters and selecting canonical representatives. This is useful for
post-processing outputs from other LLM-based extraction tools like AspectSentimentExtractor,
where near-duplicate entities are common.

The approach uses large context windows efficiently - processing up to thousands of entities
in a single LLM call, avoiding expensive recursive merging.

Example usage:

    >>> entities = [
    ...     "food too expensive", "overpriced food", "food prices high",
    ...     "long lines", "queues too long", "long wait times",
    ...     "friendly staff", "staff was nice",
    ... ]
    >>> clusterer = EntityClusterer(entities=entities)
    >>> results = await clusterer()
    >>> # Returns ClusteredEntities with clusters and canonical names
"""

import json
from collections.abc import Iterable
from functools import cached_property
from typing import ClassVar

from loguru import logger as LOG

from .. import AnyContext, Prompt, Response, ResponseClass, Tool
from ..utils import dedent


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

CLUSTER_PROMPT_SYSTEM = dedent("""
You are an expert at grouping semantically equivalent phrases.

Given a list of entities, group them into clusters where each cluster contains entities that
express the same underlying concept, complaint, or sentiment target - even if worded differently.

For each cluster, provide a canonical (representative) name that best captures the shared meaning.
The canonical name should be clear, concise, and grammatically correct.

Examples of entities that should be clustered together:
- "food too expensive", "overpriced food", "high food prices" → canonical: "expensive food"
- "long lines", "queues too long", "long wait times", "waiting too long" → canonical: "long wait times"
- "staff was rude", "unfriendly employees", "rude service" → canonical: "rude staff"

CLUSTERING GUIDELINES:
1. Merge entities that refer to the SAME specific concept (e.g., "expensive food" and "overpriced meals")
2. Keep entities SEPARATE if they refer to different aspects, even if related:
   - "expensive food" vs "expensive tickets" → SEPARATE (different targets)
   - "long ride queues" vs "long food lines" → SEPARATE (different contexts)
   - "rude staff" vs "unhelpful staff" → SEPARATE (different complaints)
3. Aim for meaningful granularity - typically 1 cluster per 5-20 input entities on average
4. Single-member clusters are fine for truly unique concepts

IMPORTANT:
- Every input entity must appear in exactly one cluster
- Be consistent: if you merge "X is expensive" with "costly X", do the same pattern throughout
- Do NOT over-merge into just a few giant clusters - preserve meaningful distinctions

${instructions}
""")

CLUSTER_PROMPT_USER = dedent("""
Group the following {{count}} entities into semantic clusters. Return ALL entities - each entity
must appear in exactly one cluster.

# Entities

{{entities}}
""")

MERGE_PROMPT_SYSTEM = dedent("""
You are an expert at identifying semantically equivalent clusters and merging them.

You will be given a list of clusters, where each cluster has a canonical name and member entities.
Some clusters from different batches may be semantically equivalent and should be merged.

Your task is to:
1. Identify clusters that are semantically equivalent (their canonical names or members overlap in meaning)
2. Merge equivalent clusters into single clusters
3. Choose the best canonical name for each merged cluster
4. Combine all members from merged clusters
5. Keep clusters that have no semantic equivalent unchanged

MERGING GUIDELINES:
1. Merge clusters ONLY if they refer to the SAME specific concept
2. Keep clusters SEPARATE if they refer to different aspects, even if related:
   - "expensive food" vs "expensive parking" → SEPARATE
   - "long wait times" vs "long ride duration" → SEPARATE
3. Preserve meaningful distinctions - don't collapse everything into a few mega-clusters
4. The goal is to consolidate TRUE duplicates, not to minimize cluster count

IMPORTANT:
- Preserve all unique member entities when merging
- If unsure whether to merge, keep separate
- The final canonical name should be the clearest, most representative option

${instructions}
""")

MERGE_PROMPT_USER = dedent("""
Review the following {{count}} clusters and merge any that are semantically equivalent.
Return ALL member entities - each must appear in exactly one cluster.

# Clusters

{{clusters}}
""")

# -----------------------------------------------------------------------------
# Response Models
# -----------------------------------------------------------------------------


class EntityCluster(Response):
    """A cluster of semantically equivalent entities."""

    canonical: str
    """The canonical/representative name for this cluster."""
    members: list[str]
    """All entities that belong to this cluster."""


class ClusteredEntities(Response):
    """Result of clustering entities into semantic groups."""

    clusters: list[EntityCluster]
    """List of entity clusters."""

    @cached_property
    def canonicals(self) -> list[str]:
        """Get all canonical names."""
        return [c.canonical for c in self.clusters]

    @cached_property
    def mapping(self) -> dict[str, str]:
        """Get a mapping from each member entity to its canonical name.

        Keys are normalized (lowercase, whitespace-collapsed) for robust matching.
        """
        return {
            _normalize(member): cluster.canonical
            for cluster in self.clusters
            for member in cluster.members
        }

    @cached_property
    def all_members(self) -> set[str]:
        """Get all member entities across all clusters (normalized)."""
        return {_normalize(member) for cluster in self.clusters for member in cluster.members}

    def coverage(self, entities: Iterable[str]) -> float:
        """Calculate what fraction of entities are covered by clusters."""
        entities_set = {_normalize(e) for e in entities}
        if not entities_set:
            return 1.0
        covered = entities_set & self.all_members
        return len(covered) / len(entities_set)

    def missing(self, entities: Iterable[str]) -> list[str]:
        """Get entities that are not in any cluster."""
        return [e for e in entities if _normalize(e) not in self.all_members]

    def to_dict(self) -> dict[str, list[str]]:
        """Convert to a dictionary mapping canonical names to members."""
        return {c.canonical: c.members for c in self.clusters}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _normalize(s: str) -> str:
    """Normalize string for pre-deduplication."""
    return " ".join(s.lower().split())


def _pre_deduplicate(entities: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Remove exact duplicates (case-insensitive) and return unique list + mapping.

    Returns:
        unique_entities: List of unique entities (first occurrence kept)
        reverse_map: Maps normalized form to all original variants
    """
    seen: dict[str, str] = {}  # normalized -> first original
    reverse_map: dict[str, list[str]] = {}  # normalized -> all originals

    for entity in entities:
        norm = _normalize(entity)
        if norm not in seen:
            seen[norm] = entity
            reverse_map[norm] = [entity]
        else:
            reverse_map[norm].append(entity)

    return list(seen.values()), reverse_map


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


class EntityClusterer(Tool):
    """Cluster semantically similar entities using LLM.

    This tool groups a list of entities into semantic clusters, where each cluster contains
    entities that express the same concept. Uses large context windows efficiently - processes
    up to thousands of entities per LLM call.

    The tool first removes exact duplicates (case-insensitive), then sends unique entities
    to the LLM for semantic clustering. If multiple batches are needed, an optional merge
    step can consolidate similar clusters across batches.

    Args:
        entities: List of entity strings to cluster
        instructions: Additional domain-specific instructions for clustering
        batch_size: Max entities per LLM call (default: 2000 - handles most cases in one call)
        merge_clusters: If True and multiple batches, merge similar clusters across batches (one LLM call)

    Example:
        >>> clusterer = EntityClusterer(
        ...     entities=["food too expensive", "overpriced food", "long lines", "queues too long"],
        ... )
        >>> results = await clusterer()
        >>> print(results.mapping)
        {'food too expensive': 'expensive food', 'overpriced food': 'expensive food', ...}
    """

    entities: Iterable[str]
    """Entities to cluster."""
    instructions: str = ""
    """Additional domain-specific instructions for the clustering task."""
    batch_size: int = 2000
    """Max entities per LLM call. Default handles most use cases in a single call."""
    merge_clusters: bool = True
    """If True, merge similar clusters (across batches or within single batch for consolidation)."""
    consolidate: bool = True
    """If True, always run a merge pass even on single-batch results to consolidate similar clusters."""

    response_model: ClassVar[ResponseClass] = ClusteredEntities

    # Internal state for tracking pre-deduplication
    _unique_entities: list[str] | None = None
    _reverse_map: dict[str, list[str]] | None = None

    def model_post_init(self, __context) -> None:
        """Pre-deduplicate entities after initialization."""
        entity_list = list(self.entities)
        self._unique_entities, self._reverse_map = _pre_deduplicate(entity_list)
        LOG.info(
            f"Pre-dedup: {len(entity_list)} entities → {len(self._unique_entities)} unique "
            f"({len(entity_list) - len(self._unique_entities)} exact duplicates removed)"
        )

    @cached_property
    def prompt(self) -> Prompt:
        prompt = Prompt(
            messages=[
                {"role": "system", "content": CLUSTER_PROMPT_SYSTEM},
                {"role": "user", "content": CLUSTER_PROMPT_USER},
            ],  # type: ignore
        )
        return prompt.substitute(instructions=self.instructions)

    @cached_property
    def context(self) -> AnyContext:
        """Create batched contexts - typically just one for most use cases."""
        entities = self._unique_entities or []
        batches = [
            entities[i : i + self.batch_size] for i in range(0, len(entities), self.batch_size)
        ]
        return [
            {
                "count": len(batch),
                "entities": "\n".join(f"- {e}" for e in batch),
            }
            for batch in batches
        ]

    def _expand_clusters(self, clusters: list[EntityCluster]) -> list[EntityCluster]:
        """Expand clusters to include all original variants from pre-deduplication."""
        if not self._reverse_map:
            return clusters

        expanded = []
        for cluster in clusters:
            all_members = []
            for member in cluster.members:
                norm = _normalize(member)
                # Add all original variants
                all_members.extend(self._reverse_map.get(norm, [member]))
            expanded.append(EntityCluster(canonical=cluster.canonical, members=all_members))
        return expanded

    def _concat_batch_results(self, results: list[ClusteredEntities]) -> ClusteredEntities:
        """Concatenate results from multiple batches without LLM merge."""
        all_clusters = []
        for result in results:
            all_clusters.extend(result.clusters)
        return ClusteredEntities(clusters=all_clusters)

    async def __call__(self, **kwargs) -> ClusteredEntities:  # type: ignore[override]
        """Run the clustering tool."""
        n_batches = len(self.context) if isinstance(self.context, list) else 1
        LOG.info(
            f"Clustering: {len(self._unique_entities or [])} entities in {n_batches} batch(es)"
        )

        result = await self.task(context=self.context, **kwargs)

        # Handle single result vs multiple batches
        if isinstance(result, ClusteredEntities):
            batch_results = [result]
        else:
            batch_results = list(result)  # type: ignore

        # Concatenate batch results
        combined: ClusteredEntities = self._concat_batch_results(batch_results)  # type: ignore
        total_members = sum(len(c.members) for c in combined.clusters)
        LOG.info(
            f"Batch results: {len(combined.clusters)} clusters, {total_members} entities covered"
        )
        LOG.info(f"Clusters: {json.dumps(combined.to_dict(), indent=2)}")

        # Optionally merge clusters (across batches or for consolidation)
        should_merge = self.merge_clusters and (len(batch_results) > 1 or self.consolidate)
        if should_merge and len(combined.clusters) > 1:
            action = "Consolidating" if len(batch_results) == 1 else "Merging"
            LOG.info(f"{action} {len(combined.clusters)} clusters...")
            merger = ClusterMerger(
                clusters=combined.clusters,
                instructions=self.instructions,
                model=self.model,
            )
            merge_result = await merger(**kwargs)
            combined = merge_result[0]  # Get ClusteredEntities from ResponseSet
            total_members_after = sum(len(c.members) for c in combined.clusters)
            LOG.info(
                f"After {action.lower()}: {len(combined.clusters)} clusters, {total_members_after} entities"
            )

        # Expand to include pre-deduplicated variants
        expanded = self._expand_clusters(combined.clusters)
        final = ClusteredEntities(clusters=expanded)
        LOG.info(
            f"Final: {len(final.clusters)} clusters, "
            f"{len(final.all_members)} unique entities mapped"
        )
        return final


class ClusterMerger(Tool):
    """Merge semantically equivalent clusters (single LLM call).

    This tool takes a list of clusters and merges those that are semantically equivalent.
    Used internally by EntityClusterer when processing multiple batches.

    Args:
        clusters: List of EntityCluster objects to merge
        instructions: Additional instructions for the merge task
    """

    clusters: list[EntityCluster]
    """Clusters to potentially merge."""
    instructions: str = ""
    """Additional domain-specific instructions."""

    response_model: ClassVar[ResponseClass] = ClusteredEntities

    @cached_property
    def prompt(self) -> Prompt:
        prompt = Prompt(
            messages=[
                {"role": "system", "content": MERGE_PROMPT_SYSTEM},
                {"role": "user", "content": MERGE_PROMPT_USER},
            ],  # type: ignore
        )
        return prompt.substitute(instructions=self.instructions)

    @cached_property
    def context(self) -> AnyContext:
        return {
            "count": len(self.clusters),
            "clusters": _format_clusters(self.clusters),
        }


def _format_clusters(clusters: list[EntityCluster]) -> str:
    """Format clusters for display in prompts."""
    lines = []
    for i, cluster in enumerate(clusters, 1):
        lines.append(f"{i}. **{cluster.canonical}**")
        for member in cluster.members:
            lines.append(f"   - {member}")
    return "\n".join(lines)


def deduplicate_entities(
    entities: Iterable[str],
    results: ClusteredEntities,
) -> list[str]:
    """Map a list of entities to their canonical forms using clustering results.

    Args:
        entities: Original list of entities (may contain duplicates)
        results: ClusteredEntities result from EntityClusterer

    Returns:
        List of canonical entity names in the same order as input.
        Entities not found in the mapping are returned as-is.
    """
    mapping = results.mapping
    return [mapping.get(_normalize(e), e) for e in entities]
