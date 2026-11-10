"""Source-preserving Fox8 account facts for evidence-first input views.

Profile fields and first-record tweets remain separately addressable. A full
view must reconstruct the 31 published model inputs; selected views rebuild
those inputs with exactly the same feature formulas. Graph relation pointers
are a separate, explicitly recorded input channel and are not edited here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .aaai_fox8_feature_schema import FEATURE_FIELDS, PROFILE_FIELDS, TWEET_FIELDS
from .aaai_fox8_raw_features import RAW_PROFILE_FIELDS, profile_features
from .aaai_fox8_native_graph import RELATIONS, sha256_file


@dataclass(frozen=True)
class TweetAtom:
    ordinal: int
    line_number: int
    tweet_id: str
    is_retweet: int
    is_reply: int
    text_length: int
    favorites: float
    retweets: float
    language: str
    created_at_unix: int | None
    mention_count: int
    hashtag_count: int
    url_count: int
    is_quote: int
    text_sha256: str
    record_sha256: str


@dataclass
class AccountAtoms:
    account_id: str
    line_number: int
    profile: dict
    tweets: tuple[TweetAtom, ...]

    @property
    def profile_field_ids(self) -> tuple[str, ...]:
        return tuple(field for field in RAW_PROFILE_FIELDS if field in self.profile)

    @property
    def tweet_ordinals(self) -> tuple[int, ...]:
        return tuple(range(len(self.tweets)))

    def source_id(self, kind: str, item: str | int) -> str:
        if kind == "profile":
            if item not in self.profile_field_ids:
                raise KeyError(item)
            return f"fox8:{self.account_id}:profile:{item}"
        if kind == "tweet":
            if item not in self.tweet_ordinals:
                raise KeyError(item)
            return f"fox8:{self.account_id}:tweet:{self.line_number}:{item}"
        raise ValueError(kind)

    def features(
        self, *, profile_fields: Iterable[str] | None = None,
        tweet_ordinals: Iterable[int] | None = None,
    ) -> np.ndarray:
        """Rebuild one account row from only the named selected source units."""
        keep_profile = (set(self.profile_field_ids) if profile_fields is None
                        else set(profile_fields))
        keep_tweets = (set(self.tweet_ordinals) if tweet_ordinals is None
                       else set(tweet_ordinals))
        if keep_profile - set(self.profile_field_ids):
            raise ValueError("selected profile source is absent")
        if keep_tweets - set(self.tweet_ordinals):
            raise ValueError("selected tweet source is absent")
        profile = {key: value for key, value in self.profile.items()
                   if key not in RAW_PROFILE_FIELDS or key in keep_profile}
        account = profile_features(profile)
        selected = [atom for atom in self.tweets if atom.ordinal in keep_tweets]
        if selected:
            n = len(selected)
            tweet = {
                "n_tweets": n,
                "retweet_ratio": sum(a.is_retweet for a in selected) / n,
                "reply_ratio": sum(a.is_reply for a in selected) / n,
                "avg_tweet_len": float(np.mean([a.text_length for a in selected])),
                "avg_favorites": float(np.mean([a.favorites for a in selected])),
                "avg_retweets": float(np.mean([a.retweets for a in selected])),
                "avg_score": float(np.mean([a.favorites + a.retweets for a in selected])),
                "unique_lang_ratio": len({a.language for a in selected}) / n,
            }
        else:
            tweet = {name: 0.0 for name in TWEET_FIELDS}
        if tuple(account) != PROFILE_FIELDS or tuple(tweet) != TWEET_FIELDS:
            raise ValueError("selected source formulas differ from feature schema")
        return np.asarray([account[name] for name in PROFILE_FIELDS] +
                          [tweet[name] for name in TWEET_FIELDS], dtype=np.float32)

    def extended_features(self, tweet_ordinals: Iterable[int] | None = None) -> dict[str, float]:
        """Additional exact facts from selected tweets; none infer text meaning."""
        keep = (set(self.tweet_ordinals) if tweet_ordinals is None
                else set(tweet_ordinals))
        if keep - set(self.tweet_ordinals):
            raise ValueError("selected tweet source is absent")
        chosen = [atom for atom in self.tweets if atom.ordinal in keep]
        if not chosen:
            return {name: 0.0 for name in EXTENDED_TWEET_FIELDS}
        n = len(chosen)
        times = sorted(atom.created_at_unix for atom in chosen
                       if atom.created_at_unix is not None)
        gaps = np.diff(times) / 3600 if len(times) >= 2 else np.asarray([])
        languages = {}
        for atom in chosen:
            languages[atom.language] = languages.get(atom.language, 0) + 1
        probs = np.asarray(list(languages.values()), dtype=np.float64) / n
        values = {
            "observed_span_hours": ((times[-1] - times[0]) / 3600
                                    if len(times) >= 2 else 0.0),
            "median_intertweet_gap_hours": float(np.median(gaps)) if len(gaps) else 0.0,
            "gap_cv": (float(np.std(gaps) / np.mean(gaps))
                       if len(gaps) >= 2 and np.mean(gaps) > 0 else 0.0),
            "mention_rate": sum(atom.mention_count for atom in chosen) / n,
            "hashtag_rate": sum(atom.hashtag_count for atom in chosen) / n,
            "url_rate": sum(atom.url_count for atom in chosen) / n,
            "quote_ratio": sum(atom.is_quote for atom in chosen) / n,
            "exact_text_duplicate_ratio": 1 - len({atom.text_sha256 for atom in chosen}) / n,
            "language_entropy": float(-np.sum(probs * np.log(probs))),
        }
        if tuple(values) != EXTENDED_TWEET_FIELDS:
            raise ValueError("extended fact order differs")
        return values


@dataclass(frozen=True)
class RelationAtom:
    source_id: str
    source_account_id: str
    target_account_id: str
    relation: str
    raw_locators: tuple[tuple[int, int, str], ...]


def load_relation_atoms(
    event_index_path: Path, raw_path: Path, account_ids: Iterable[str],
) -> tuple[RelationAtom, ...]:
    """Keep each released native relation event as an addressable source unit."""
    payload = json.loads(event_index_path.read_text(encoding="utf-8"))
    if payload["raw_source_sha256"] != sha256_file(raw_path):
        raise ValueError("relation index differs from raw Fox8 source")
    allowed = set(account_ids)
    atoms = []
    seen = set()
    for event in payload["events"]:
        source = str(event["source_id"])
        target = str(event["target_id"])
        relation = event["relation"]
        if relation not in ("reply", "retweet"):
            raise ValueError("unexpected relation type")
        if source not in allowed or target not in allowed or source == target:
            continue
        locators = tuple((int(item["line"]), int(item["tweet_index"]),
                          str(item["field"])) for item in event["locators"])
        if not locators or len(locators) != int(event["count"]):
            raise ValueError("relation pointer multiplicity differs")
        source_id = f"fox8:relation:{source}:{relation}:{target}"
        if source_id in seen:
            raise ValueError("duplicate native relation source")
        seen.add(source_id)
        atoms.append(RelationAtom(source_id, source, target, relation, locators))
    return tuple(atoms)


def graph_from_relation_atoms(
    relations: Iterable[RelationAtom], account_ids: Iterable[str],
    selected_source_ids: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Construct both directions of the typed graph from selected events."""
    ids = tuple(account_ids)
    index = {uid: i for i, uid in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("graph account IDs must be unique")
    keep = None if selected_source_ids is None else set(selected_source_ids)
    pairs: dict[str, set[tuple[int, int]]] = {name: set() for name in RELATIONS}
    observed = set()
    for item in relations:
        observed.add(item.source_id)
        if keep is not None and item.source_id not in keep:
            continue
        source = index[item.source_account_id]
        target = index[item.target_account_id]
        pairs[item.relation].add((source, target))
        pairs[f"rev_{item.relation}"].add((target, source))
    if keep is not None and keep - observed:
        raise ValueError("selected relation source is absent")
    return {
        name: (torch.tensor(sorted(pairs[name]), dtype=torch.long).T.contiguous()
               if pairs[name] else torch.empty((2, 0), dtype=torch.long))
        for name in RELATIONS
    }


EXTENDED_TWEET_FIELDS = (
    "observed_span_hours", "median_intertweet_gap_hours", "gap_cv",
    "mention_rate", "hashtag_rate", "url_rate", "quote_ratio",
    "exact_text_duplicate_ratio", "language_entropy",
)


def _unix_timestamp(value: object) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return int(datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y").timestamp())
    except ValueError:
        return None


def load_account_atoms(raw_path: Path, account_ids: Iterable[str]) -> dict[str, AccountAtoms]:
    """Read first released account records without loading tweet text in memory."""
    wanted = set(account_ids)
    accounts: dict[str, AccountAtoms] = {}
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid not in wanted or uid in accounts:
                continue
            raw_tweets = row["user_tweets"]
            if not raw_tweets:
                raise ValueError(f"account {uid} has no tweets")
            profile = raw_tweets[0]["user"]
            if str(profile.get("id_str", profile.get("id"))) != uid:
                raise ValueError(f"account {uid} profile identity differs")
            atoms = []
            for ordinal, item in enumerate(raw_tweets):
                tweet_text = item.get("text", "") or ""
                entities = item.get("entities") or {}
                raw_record = json.dumps(item, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":"))
                atoms.append(TweetAtom(
                    ordinal=ordinal, line_number=line_number,
                    tweet_id=str(item.get("id_str", item.get("id", ""))),
                    is_retweet=int(tweet_text.startswith("RT ")),
                    is_reply=int(item.get("in_reply_to_status_id") is not None),
                    text_length=len(tweet_text),
                    favorites=item.get("favorite_count", 0) or 0,
                    retweets=item.get("retweet_count", 0) or 0,
                    language=item.get("lang", "und"),
                    created_at_unix=_unix_timestamp(item.get("created_at")),
                    mention_count=len(entities.get("user_mentions") or []),
                    hashtag_count=len(entities.get("hashtags") or []),
                    url_count=len(entities.get("urls") or []),
                    is_quote=int(bool(item.get("is_quote_status", False))),
                    text_sha256=sha256(tweet_text.encode("utf-8")).hexdigest(),
                    record_sha256=sha256(raw_record.encode("utf-8")).hexdigest(),
                ))
            accounts[uid] = AccountAtoms(uid, line_number, dict(profile), tuple(atoms))
            if len(accounts) == len(wanted):
                break
    if set(accounts) != wanted:
        raise ValueError(f"raw data lacks {len(wanted - set(accounts))} accounts")
    return accounts
