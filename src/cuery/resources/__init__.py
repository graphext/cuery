from os.path import dirname, join

import pandas as pd


def path(*paths):
    """Get full path to local resource."""
    return join(dirname(__file__), *paths)


def google_langs():
    df = pd.read_csv(path("google_ads_langs.csv"))
    df["Language name"] = df["Language name"].str.lower()
    return df


def google_countries():
    df = pd.read_csv(path("google_ads_countries.csv"))
    df["Name"] = df["Name"].str.lower()
    df["Country Code"] = df["Country Code"].str.lower()
    df["Criteria ID"] = df["Criteria ID"].astype(str)
    return df


GOOGLE_ADS_LANGS = google_langs()
GOOGLE_ADS_COUNTRIES = google_countries()


def google_lang_id(lang: str) -> int:
    """Get Google Ads language ID from language name."""
    lang = lang.lower()

    if lang in GOOGLE_ADS_LANGS["Language name"].values:  # noqa: PD011
        return GOOGLE_ADS_LANGS.loc[
            GOOGLE_ADS_LANGS["Language name"] == lang, "Criterion ID"
        ].item()

    if lang in GOOGLE_ADS_LANGS["Language code"].values:  # noqa: PD011
        return GOOGLE_ADS_LANGS.loc[
            GOOGLE_ADS_LANGS["Language code"] == lang, "Criterion ID"
        ].item()

    raise ValueError(f"Language '{lang}' not found in Google Ads languages.")


def google_country_id(country: str) -> str:
    """Get Google Ads country ID from country name or code."""
    country = country.lower()

    if country in GOOGLE_ADS_COUNTRIES["Name"].values:  # noqa: PD011
        return GOOGLE_ADS_COUNTRIES.loc[
            GOOGLE_ADS_COUNTRIES["Name"] == country, "Criteria ID"
        ].item()

    if country in GOOGLE_ADS_COUNTRIES["Country Code"].values:  # noqa: PD011
        return GOOGLE_ADS_COUNTRIES.loc[
            GOOGLE_ADS_COUNTRIES["Country Code"] == country, "Criteria ID"
        ].item()

    raise ValueError(f"Country '{country}' not found in Google Ads countries.")
