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
from typing import ClassVar, Self

from loguru import logger as LOG
from pydantic import model_validator

from .. import AnyContext, Prompt, Response, ResponseClass, Tool
from ..task import Task
from ..utils import dedent


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

CLUSTER_PROMPT_SYSTEM = dedent("""
You are an expert at grouping semantically equivalent phrases for deduplication.

Given a list of entities, group them into clusters where each cluster contains entities that
express the same underlying concept - even if worded differently. The goal is DEDUPLICATION:
reducing redundant phrases to a smaller set of canonical forms.

For each cluster, provide a canonical (representative) name that best captures the shared meaning.
The canonical name should be clear, concise, and grammatically correct.

EXAMPLES OF ENTITIES TO CLUSTER TOGETHER:
- "food too expensive", "overpriced food", "high food prices", "costly meals" → canonical: "expensive food"
- "long lines", "queues too long", "long wait times", "waiting forever" → canonical: "long wait times"
- "staff was rude", "unfriendly employees", "rude service", "impolite workers" → canonical: "rude staff"
- "excellent ride", "great ride", "amazing ride", "fantastic ride" → canonical: "excellent ride"
- "poor food quality", "bad food", "terrible food", "food was awful" → canonical: "poor food quality"

WHEN TO MERGE (same cluster):
- Same target + same sentiment: "expensive food" = "overpriced food" = "food costs too much"
- Synonyms: "long wait" = "long queue" = "waiting forever"
- Spelling/phrasing variations: "crowded park" = "park too crowded" = "overcrowded park"

WHEN TO KEEP SEPARATE (different clusters):
- Different targets: "expensive food" ≠ "expensive tickets" (different things being expensive)
- Different sentiments: "excellent food" ≠ "poor food" (opposite sentiments)
- Genuinely different concepts: "rude staff" ≠ "slow service" (different complaints)

CLUSTERING TARGETS:
- Aim to REDUCE the entity count significantly - typically 3-10 input entities per cluster on average
- Most clusters should have 2+ members; single-member clusters only for truly unique concepts
- If you see many similar phrasings, merge them aggressively

NEVER DO THIS:
- NEVER create catch-all clusters like "general feedback", "all feedback", "miscellaneous", "other"
- NEVER put more than 50 entities in a single cluster
- NEVER create single-member clusters for entities that have synonyms in the list
- NEVER give up - process ALL entities with equal care

${instructions}
""")

CLUSTER_PROMPT_USER = dedent("""
Group the following {{count}} entities into semantic clusters for deduplication.
Merge aggressively - if two phrases mean the same thing, they belong together.
Return ALL entities - each must appear in exactly one cluster.

# Entities

{{entities}}
""")

MERGE_PROMPT_SYSTEM = dedent("""
You are an expert at identifying semantically equivalent clusters.

You will be given a list of clusters in JSON format, where each key is the canonical name and
the value is a sample of member entities. Your task is to identify which clusters should be
merged because they represent the same underlying concept.

For each group of clusters that should be merged:
1. Choose the best canonical name to keep
2. List the other canonical names that should be merged into it

MERGING GUIDELINES:
1. Merge clusters ONLY if they refer to the SAME specific concept
2. Keep clusters SEPARATE if they refer to different aspects, even if related:
   - "expensive food" vs "expensive parking" → SEPARATE (different targets)
   - "long wait times" vs "long ride duration" → SEPARATE (different things)
   - "rude staff" vs "unhelpful staff" → SEPARATE (different complaints)
3. The goal is to consolidate TRUE duplicates, not to minimize cluster count
4. When in doubt, keep clusters separate
5. Only output clusters that need merging - skip clusters that should remain standalone

EXAMPLE:

Input clusters:
{
  "expensive food": ["food too expensive", "overpriced food"],
  "high food prices": ["high prices for food", "costly meals"],
  "long wait times": ["waiting too long", "long queues"],
  "long ride duration": ["rides are too short", "brief ride times"],
  "rude staff": ["staff was rude", "unfriendly employees"]
}

Correct output:
{
  "groups": [
    {"canonical": "expensive food", "merge": ["high food prices"]}
  ]
}

Explanation:
- "expensive food" and "high food prices" → MERGE (same concept: food is too expensive)
- "long wait times" and "long ride duration" → SEPARATE (different: waiting in line vs ride length)
- "rude staff" → no merge candidates, so not included in output

${instructions}
""")

MERGE_PROMPT_USER = dedent("""
Review the following {{count}} clusters and identify which ones should be merged.

Only output groups of clusters that should be merged together. For each group, specify:
- "canonical": The canonical name to keep (the best representative)
- "merge": List of other canonical names to merge into it

If no clusters should be merged, return {"groups": []}.

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

    # Class variable for max cluster size validation (can be set dynamically)
    _max_cluster_size: ClassVar[int | None] = None
    _total_entities: ClassVar[int | None] = None

    @model_validator(mode="after")
    def validate_no_degenerate_clusters(self) -> Self:
        """Reject catch-all clusters and other degenerate patterns."""
        # Check for catch-all cluster names
        catch_all_names = {
            "all feedback",
            "all_feedback",
            "general feedback",
            "general_feedback",
            "miscellaneous",
            "misc",
            "other",
            "various",
            "various issues",
            "all issues",
            "all_issues",
            "everything",
            "all",
            "placeholder",
        }
        for cluster in self.clusters:
            if cluster.canonical.lower().strip() in catch_all_names:
                raise ValueError(
                    f"Catch-all cluster '{cluster.canonical}' is not allowed. "
                    f"Create specific clusters instead."
                )

        # Check for empty clusters
        for cluster in self.clusters:
            if not cluster.members:
                raise ValueError(f"Empty cluster '{cluster.canonical}' is not allowed.")

        # Check for oversized clusters (if limit is set)
        max_size = self._max_cluster_size
        if max_size is not None:
            for cluster in self.clusters:
                if len(cluster.members) > max_size:
                    raise ValueError(
                        f"Cluster '{cluster.canonical}' has {len(cluster.members)} members, "
                        f"exceeding max of {max_size}. Split into more specific clusters."
                    )

        # Check that we don't have too few clusters relative to entity count (over-merging)
        total = self._total_entities
        if total is not None and total > 20:
            min_clusters = max(2, total // 50)
            if len(self.clusters) < min_clusters:
                raise ValueError(
                    f"Only {len(self.clusters)} clusters for {total} entities is too few. "
                    f"Expected at least {min_clusters} clusters. Create more specific groupings."
                )

        # Check for too many single-member clusters (under-clustering)
        if total is not None and total > 20:
            single_member_clusters = sum(1 for c in self.clusters if len(c.members) == 1)
            single_member_ratio = (
                single_member_clusters / len(self.clusters) if self.clusters else 0
            )
            # If more than 60% of clusters have only 1 member, that's too atomic
            if single_member_ratio > 0.6 and single_member_clusters > 10:
                raise ValueError(
                    f"Too many single-member clusters: {single_member_clusters} of {len(self.clusters)} "
                    f"({single_member_ratio:.0%}). Merge similar entities more aggressively."
                )

        return self

    @classmethod
    def with_validation_limits(
        cls, max_cluster_size: int | None = None, total_entities: int | None = None
    ) -> type["ClusteredEntities"]:
        """Create a subclass with validation limits baked in."""

        class BoundClusteredEntities(cls):
            _max_cluster_size: ClassVar[int | None] = max_cluster_size
            _total_entities: ClassVar[int | None] = total_entities

        return BoundClusteredEntities

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

    @property
    def member_count(self) -> int:
        """Get the total number of member entities across all clusters."""
        return sum(len(c.members) for c in self.clusters)

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


class MergeGroup(Response):
    """A group of clusters that should be merged together."""

    canonical: str
    """The canonical name to keep (best representative for the merged cluster)."""
    merge: list[str]
    """Other canonical names that should be merged into this cluster."""

    @model_validator(mode="after")
    def validate_no_self_reference(self) -> Self:
        """Ensure canonical name is not in its own merge list."""
        if self.canonical in self.merge:
            raise ValueError(
                f"Canonical name '{self.canonical}' cannot appear in its own merge list"
            )
        return self


class MergeInstructions(Response):
    """Instructions for which clusters to merge."""

    groups: list[MergeGroup]
    """Groups of clusters to merge. Each group specifies a canonical to keep and others to merge into it."""

    # Class variable to hold valid canonicals for validation (set dynamically)
    _valid_canonicals: ClassVar[set[str] | None] = None

    @model_validator(mode="after")
    def validate_merge_instructions(self) -> Self:
        """Validate merge instructions for consistency and against valid canonicals."""
        seen_canonicals: set[str] = set()
        seen_merges: set[str] = set()
        valid = self._valid_canonicals  # May be None if not using dynamic validation

        for group in self.groups:
            # Check canonical isn't used multiple times
            if group.canonical in seen_canonicals:
                raise ValueError(f"Canonical name '{group.canonical}' appears in multiple groups")
            seen_canonicals.add(group.canonical)

            # Check canonical exists in valid set (if provided)
            if valid is not None and group.canonical not in valid:
                raise ValueError(
                    f"Unknown canonical name '{group.canonical}' - not in original clusters"
                )

            # Check merge targets aren't duplicated and exist
            for name in group.merge:
                if name in seen_merges:
                    raise ValueError(f"Name '{name}' appears in multiple merge lists")
                if name in seen_canonicals:
                    raise ValueError(f"Name '{name}' is both a canonical and a merge target")
                seen_merges.add(name)

                # Check merge target exists in valid set (if provided)
                if valid is not None and name not in valid:
                    raise ValueError(f"Unknown merge name '{name}' - not in original clusters")

        return self

    @classmethod
    def with_valid_canonicals(cls, valid_canonicals: set[str]) -> type["MergeInstructions"]:
        """Create a subclass with valid canonicals baked in for validation.

        This allows validation to happen during Pydantic parsing, triggering
        LLM retries on invalid responses.

        Args:
            valid_canonicals: Set of valid canonical names from the original clusters.

        Returns:
            A dynamically created MergeInstructions subclass with validation.
        """

        class BoundMergeInstructions(cls):
            _valid_canonicals: ClassVar[set[str] | None] = valid_canonicals

        return BoundMergeInstructions


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
    max_cluster_size: int = 100
    """Maximum allowed members per cluster. Larger clusters trigger validation error and retry."""

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
    def response_model(self) -> ResponseClass:
        """Create response model with validation limits for cluster size."""
        n_entities = len(self._unique_entities or [])
        return ClusteredEntities.with_validation_limits(
            max_cluster_size=self.max_cluster_size,
            total_entities=n_entities,
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
            LOG.info(f"{action} {len(combined.clusters)} clusters ({total_members} entities)...")
            merger = ClusterMerger(
                clusters=combined.clusters,
                instructions=self.instructions,
                model=self.model,
            )
            # ClusterMerger applies merges programmatically - no data loss possible
            merged = await merger(**kwargs)
            total_members_after = merged.member_count

            # Log merged clusters for debugging
            LOG.info(f"Merged clusters: {json.dumps(merged.to_dict(), indent=2)}")
            LOG.info(
                f"After {action.lower()}: {len(merged.clusters)} clusters, {total_members_after} entities"
            )
            combined = merged

        # Expand to include pre-deduplicated variants
        expanded = self._expand_clusters(combined.clusters)
        final = ClusteredEntities(clusters=expanded)
        LOG.info(
            f"Final: {len(final.clusters)} clusters, "
            f"{len(final.all_members)} unique entities mapped"
        )
        return final


class ClusterMerger(Tool):
    """Merge semantically equivalent clusters using LLM-guided instructions.

    This tool asks the LLM to identify which clusters should be merged (by canonical name),
    then applies the merges programmatically. This approach:
    - Never loses entities (merging is done in code, not by LLM)
    - Requires much smaller LLM output (just canonical names, not all entities)
    - Is more reliable than asking LLM to output all entities again

    Args:
        clusters: List of EntityCluster objects to merge
        instructions: Additional instructions for the merge task
    """

    clusters: list[EntityCluster]
    """Clusters to potentially merge."""
    instructions: str = ""
    """Additional domain-specific instructions."""

    # Note: response_model is set dynamically via property to include valid canonicals

    @cached_property
    def response_model(self) -> ResponseClass:
        """Create response model with valid canonicals baked in for validation."""
        valid_canonicals = {c.canonical for c in self.clusters}
        return MergeInstructions.with_valid_canonicals(valid_canonicals)

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
        # Just show canonical names with a sample of members for context
        cluster_info = {}
        for c in self.clusters:
            # Show up to 5 members as examples
            sample = c.members[:5]
            if len(c.members) > 5:
                sample.append(f"... and {len(c.members) - 5} more")
            cluster_info[c.canonical] = sample
        return {
            "count": len(self.clusters),
            "clusters": json.dumps(cluster_info, indent=2),
        }

    def _apply_merge_instructions(self, instructions: MergeInstructions) -> ClusteredEntities:
        """Apply merge instructions to clusters programmatically."""
        # Build map of canonical name -> cluster
        cluster_map: dict[str, EntityCluster] = {c.canonical: c for c in self.clusters}

        # Track which clusters have been merged away
        merged_away: set[str] = set()

        # Process each merge group
        for group in instructions.groups:
            if group.canonical not in cluster_map:
                LOG.warning(f"Merge target '{group.canonical}' not found in clusters, skipping")
                continue

            # Collect all members from clusters to merge
            target_cluster = cluster_map[group.canonical]
            all_members = list(target_cluster.members)

            for source_name in group.merge:
                if source_name in merged_away:
                    LOG.warning(f"'{source_name}' already merged, skipping")
                    continue
                if source_name not in cluster_map:
                    LOG.warning(f"Merge source '{source_name}' not found in clusters, skipping")
                    continue

                source_cluster = cluster_map[source_name]
                all_members.extend(source_cluster.members)
                merged_away.add(source_name)

            # Update the target cluster with combined members
            cluster_map[group.canonical] = EntityCluster(
                canonical=group.canonical,
                members=all_members,
            )

        # Build final list: all clusters except those merged away
        final_clusters = [
            cluster for name, cluster in cluster_map.items() if name not in merged_away
        ]

        return ClusteredEntities(clusters=final_clusters)

    async def __call__(self, **kwargs) -> ClusteredEntities:  # type: ignore[override]
        """Get merge instructions from LLM and apply them programmatically."""
        # Get merge instructions from LLM (validation happens during parsing, triggers retries)
        result = await self.task(context=self.context, **kwargs)
        instructions: MergeInstructions = result[0]  # type: ignore

        # Log what merges were requested
        if instructions.groups:
            merge_summary = {g.canonical: g.merge for g in instructions.groups}
            LOG.info(f"Merge instructions: {json.dumps(merge_summary, indent=2)}")
        else:
            LOG.info("No merges suggested by LLM")

        # Apply merges programmatically (guaranteed to preserve all entities)
        return self._apply_merge_instructions(instructions)


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
