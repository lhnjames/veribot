"""Named Fox8 account features used as inputs to the VeriBot graph branches."""

PROFILE_FIELDS = (
    "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count",
    "verified", "default_profile", "default_profile_image", "protected", "geo_enabled",
    "profile_use_background_image", "has_extended_profile", "contributors_enabled",
    "is_translator", "is_translation_enabled", "profile_background_tile", "description_len",
    "name_len", "screen_name_len", "has_location", "has_url", "has_description", "ff_ratio",
)

TWEET_FIELDS = (
    "n_tweets", "retweet_ratio", "reply_ratio", "avg_tweet_len", "avg_favorites",
    "avg_retweets", "avg_score", "unique_lang_ratio",
)

FEATURE_FIELDS = PROFILE_FIELDS + TWEET_FIELDS
