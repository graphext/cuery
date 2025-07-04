"""Use Apify actors to fetch domain traffic data from Similarweb and other sources."""

from pandas import DataFrame


def clean_domain(domain: str) -> str | None:
    """Clean domain name."""
    dot_coms = ["X", "youtube", "reddit", "facebook", "instagram", "twitter", "linkedin", "tiktok"]
    if domain.lower() in dot_coms:
        return domain.lower() + ".com"

    if not domain.startswith("http"):
        return None

    return domain


def fetch_traffic(df: DataFrame) -> DataFrame:
    """Fetch traffic data for a DataFrame of organic SERP results."""
    domains = [d for dlist in df.domains.dropna() for d in dlist]
    domains = list(set(domains))
    domains = [cd for d in domains if (cd := clean_domain(d))]
    ...  # noqa: PIE790

    return df
