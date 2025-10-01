"""Module for handling SEO source categorization and related utilities."""

from numpy import ndarray
from pandas import DataFrame, Series

from ..tools import TopicAssigner
from ..utils import LOG, dedent

CATERORIES_3 = {
    "Authority & Ownership": {
        "Brand / Corporate": ["Company homepage", "Brand sites", "Product microsites"],
        "Government / Public Sector": [
            "Ministries",
            "Municipalities",
            "Agencies",
            "Official portals",
        ],
        "Education / Academic": [
            "Universities",
            "Schools",
            "Research centers",
            "Academic publishers",
        ],
        "Non-Profit / NGO": ["Charities", "Advocacy groups", "Think tanks"],
        "Religious / Faith-Based": ["Churches", "Temples", "Religious organizations"],
    },
    "Media & Information": {
        "News & Journalism": [
            "Mainstream news outlets",
            "Trade & niche publications",
            "Local & regional news",
            "Press release distribution",
        ],
        "Reference & Knowledge": [
            "Encyclopedic / general reference",
            "Data & statistics portals",
            "Guides & tutorials",
            "Educational content hubs",
        ],
        "Content & Community": [
            "Blogs (personal, expert, niche)",
            "Forums & message boards",
            "Social media platforms",
            "Aggregators & Q&A (Reddit, Quora, Hacker News)",
            "User review sites (Trustpilot, Yelp, G2)",
        ],
    },
    "Commercial & Services": {
        "E-Commerce & Marketplaces": [
            "Retailers & general marketplaces",
            "Affiliate & comparison sites",
            "Classifieds & listings",
            "Travel & booking platforms",
            "Job boards & recruitment",
        ],
        "Professional & B2B": [
            "Consultancies & agencies",
            "Professional services (legal, medical, financial)",
            "SaaS product sites",
            "B2B marketplaces & vendor directories",
        ],
        "Technology & Tools": [
            "Developer documentation (APIs, GitHub, OSS docs)",
            "Online calculators & utilities",
            "Tech blogs & knowledge hubs",
        ],
    },
    "Lifestyle & Entertainment": {
        "Media & Entertainment": [
            "Streaming platforms (video, music, podcasts)",
            "Sports & gaming sites",
        ],
        "Lifestyle & Culture": ["Food, travel, fashion, hobbies, wellness"],
        "Events & Ticketing": ["Event listings", "Ticket sales", "Cultural events"],
    },
    "Low-Value / Edge Cases": {
        "Parked / Placeholder Domains": [],
        "Spam / Low-Quality SEO Sites": [],
        "Personal / Portfolio Sites": [],
    },
}

CATEGORIES_2 = {
    "Authority & Ownership": [
        "Brand / Corporate Site",
        "Product / Microsite",
        "Government / Public Sector",
        "Education / Academic",
        "Non-Profit / NGO",
        "Religious / Faith-Based",
    ],
    "News & Media": [
        "Mainstream News Outlet",
        "Trade / Niche Publication",
        "Local / Regional News",
        "Press Release Distribution",
    ],
    "Reference & Knowledge": [
        "Encyclopedic Reference",
        "Data / Statistics Portal",
        "Guides & Tutorials",
        "Educational Content Hub",
    ],
    "Content & Community": [
        "Blog",
        "Forum / Message Board",
        "Social Network",
        "Aggregator / Q&A Platform",
        "User Review Site",
    ],
    "Commercial": [
        "E-Commerce / Retailer",
        "Marketplace",
        "Affiliate / Comparison Site",
        "Classifieds / Listings",
        "Travel / Booking Platform",
        "Job Board / Recruitment",
    ],
    "Professional & B2B": [
        "Consultancy / Agency",
        "Professional Services (Legal, Medical, Financial)",
        "SaaS Product Site",
        "B2B Marketplace / Vendor Directory",
        "Technology Documentation (APIs, GitHub, OSS)",
        "Online Tool / Calculator",
        "Tech Blog / Knowledge Hub",
    ],
    "Lifestyle & Entertainment": [
        "Streaming Platform",
        "Sports Site / Gaming Site",
        "Lifestyle & Culture (Food, Fashion, Travel, Hobbies)",
        "Events & Ticketing",
    ],
    "Low-Value / Edge Cases": [
        "Parked / Placeholder Domain",
        "Spam / Low-Quality SEO Site",
        "Personal / Portfolio Site",
    ],
}

INSTRUCTIONS = dedent("""
You'll receive web site URLs (domains) extracted from sources cited by LLMs.
Your task is to categorize each domain/url into one of the predefined categories/topics.
""")


def find_all_strings(
    df: DataFrame,
    keys: list[str] | None = None,
    unique: bool = False,
    to_pandas: bool = True,
) -> list[str] | Series:
    """Extract all urls from a DataFrame's columns containing url lists.

    For each column, checks the first valid row to determine if it contains scalar values,
    lists of scalars or lists of dicts with any of the `keys`. Collects all scalars
    across these columns and returns them as a (unique) list.
    """
    if keys is None:
        keys = ["url", "domain"]

    values = []
    for col in df.columns:
        for value in df[col].dropna():
            if isinstance(value, str) and value:
                values.append(value)
            elif isinstance(value, list | ndarray):
                for item in value:
                    if isinstance(item, str):
                        values.append(item)
                    elif isinstance(item, dict):
                        for key in keys:
                            if key in item and (v := item[key]) and isinstance(v, str):
                                values.append(item[key])

    if unique:
        values = list(set(values))

    if to_pandas:
        return Series(values, name="domain")

    return values


async def categorize(
    domains: DataFrame | Series | list[str],
    attrs: list[str] | None = None,
    model: str = "openai/gpt-4.1-mini",
    n_concurrent: int = 100,
    **kwds,
) -> DataFrame:
    """Categorize a list of domains/URLs into predefined SEO source categories."""
    if isinstance(domains, list):
        domains = Series(domains, name="domain")
    if isinstance(domains, Series):
        domains = domains.to_frame()

    assigner = TopicAssigner(
        model=model,
        topics=CATEGORIES_2,  # type: ignore
        records=domains,
        attrs=attrs,
        instructions=INSTRUCTIONS,
    )
    return await assigner(model=model, n_concurrent=n_concurrent, **kwds)


def mapper(
    categorization: DataFrame,
    as_tuples: bool = False,
) -> dict[str, dict[str, str]] | dict[str, tuple[str, str]]:
    """Create a mapping dictionary from domain to its category and subcategory."""
    domain_mapper = {}
    for row in categorization.itertuples():
        if as_tuples:
            domain_mapper[row.domain] = (row.topic, row.subtopic)
        else:
            domain_mapper[row.domain] = {"category": row.topic, "subcategory": row.subtopic}

    return domain_mapper


def flat_domains(sources: list[dict] | None, with_subdomain: bool = True) -> list[str]:
    """Extract and flatten domains from a list of source dictionaries."""
    if not sources:
        return []

    return [domain for src in sources if (domain := src.get("domain"))]


def source_domains(sources: Series, with_subdomain: bool = True) -> Series:
    """Extract and flatten domains from a Series of lists of source dictionaries."""
    ser = sources.apply(lambda x: flat_domains(x, with_subdomain=with_subdomain))
    ser.name = sources.name + "_domains"  # type: ignore
    return ser


def map_domains(domains: Series, mapper: dict) -> tuple[Series, Series]:
    """Map a Series of lists of domains to their categories using the provided mapping dictionary.

    Returns two series, one with lists of categories, one with lists of subcategories.
    """

    def map_categories(domains):
        categories = []
        subcategories = []
        for domain in domains:
            if category := mapper.get(domain):
                categories.append(category["category"])
                subcategories.append(category["subcategory"])
            else:
                categories.append(None)
                subcategories.append(None)
        return categories, subcategories

    mapped = domains.apply(map_categories)
    cats, subcats = list(zip(*mapped, strict=True))
    cats = Series(cats, name=str(domains.name) + "_categories")
    subcats = Series(subcats, name=str(domains.name) + "_subcategories")

    return cats, subcats


def enrich_sources(ser: Series, mapper: dict):
    """Enrich a Series of source lists with their categorized domains in-place(!)."""
    for i in range(len(ser)):
        sources = ser.iloc[i]
        if not sources:
            continue

        for source in sources:
            if isinstance(source, dict) and (domain := source.get("domain")):
                categories = mapper.get(domain)
                if categories:
                    source["category"] = categories["category"]
                    source["subcategory"] = categories["subcategory"]


async def process_sources(
    df: DataFrame,
    models: list[str],
    domain_mapper: dict | None = None,
) -> DataFrame:
    """Process and enrich DataFrame columns containing source lists with categorized domains."""
    source_cols = [col for m in models if (col := f"sources_{m}") in df]

    if source_cols:
        LOG.info(f"Analysing following source columns: {source_cols}")
        LOG.info(df[source_cols])
    else:
        LOG.warning("No source columns found in DataFrame, skipping source processing.")
        return df

    # Flat lists of cited domains for each model
    for source_col in source_cols:
        df[f"{source_col}_domains"] = source_domains(df[source_col])

    # All unique domains across all models
    if domain_mapper is None:
        LOG.info("Extracting and categorizing all unique cited domains...")
        domains = find_all_strings(df[source_cols], keys=["domain"], unique=True)
        domain_categories = await categorize(domains)
        domain_mapper = mapper(domain_categories)

    # Enrich original sources columns in-place(!)
    for col in source_cols:
        enrich_sources(df[col], mapper=domain_mapper)

    # Add categories and subcategories columns for each model's domains
    domain_cols = [f"{col}_domains" for col in source_cols]
    for col in domain_cols:
        cats, subcats = map_domains(df[col], mapper=domain_mapper)
        df[f"{col}_categories"] = cats
        df[f"{col}_subcategories"] = subcats

    # If more than one model, combine all flat domains into a single domains column
    # I.e. simply concatenate the domain lists in each row (not unique, and handling None)
    if len(domain_cols) > 1:
        df["sources_domains"] = df[domain_cols].apply(
            lambda row: [domain for lst in row if isinstance(lst, list) for domain in lst],
            axis=1,
        )

    return df
