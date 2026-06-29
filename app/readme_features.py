"""README feature extraction.

Pure, deterministic functions that turn raw README markdown into numeric
features. Shared by the data collector (training) and the inference app so the
exact same logic runs at train and serving time (no skew).
"""
import re

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoji extensions
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002190-\U000021FF"  # arrows
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "]"
)

# Buzzwords devs will recognise themselves in. Matched as whole-ish substrings,
# case-insensitive. The painful ones.
BUZZWORDS = [
    "blazingly fast", "blazing fast", "lightning fast", "production-ready",
    "production ready", "battle-tested", "battle tested", "lightweight",
    "zero-config", "zero config", "zero-dependency", "batteries included",
    "out of the box", "out-of-the-box", "cutting-edge", "cutting edge",
    "state-of-the-art", "state of the art", "next-generation", "next generation",
    "seamless", "effortless", "elegant", "beautiful", "powerful", "robust",
    "scalable", "modern", "simple", "intuitive", "feature-rich", "drop-in",
    "plug-and-play", "supercharge", "turbocharge", "rock-solid", "first-class",
    "world-class", "enterprise-grade", "fully-featured",
]

# Signals of unfinished / aspirational work.
WIP_MARKERS = [
    "coming soon", "work in progress", "work-in-progress", "wip",
    "under construction", "todo", "fixme", "not yet implemented",
    "to be implemented", "stay tuned", "roadmap", "planned features",
]

BADGE_HOSTS = ("shields.io", "badgen.net", "badge.fury.io", "travis-ci",
               "circleci.com", "codecov.io", "coveralls.io", "github.com/.*/actions",
               "img.shields", "forthebadge.com", "herokucdn", "app.netlify.com")


def _count_any(text_lower, needles):
    return sum(text_lower.count(n) for n in needles)


def extract(readme: str) -> dict:
    """Return a flat dict of numeric README features.

    Always returns the same keys, even for empty/None input, so the schema is
    stable across rows.
    """
    text = readme or ""
    low = text.lower()
    n_chars = len(text)
    words = re.findall(r"\b\w+\b", text)
    n_words = len(words) or 1  # avoid div-by-zero in densities
    lines = text.splitlines()

    emojis = EMOJI_RE.findall(text)
    n_emoji = len(emojis)
    rocket = text.count("\U0001F680")  # 🚀
    fire = text.count("\U0001F525")    # 🔥
    sparkles = text.count("\U00002728")  # ✨

    md_images = re.findall(r"!\[[^\]]*\]\([^)]+\)", text)
    html_images = re.findall(r"<img\b", low)
    n_images = len(md_images) + len(html_images)
    badge_count = sum(
        1 for img in md_images for h in BADGE_HOSTS if re.search(h, img.lower())
    ) + sum(1 for h in BADGE_HOSTS if re.search(h, low) and "<img" in low)

    md_links = re.findall(r"(?<!\!)\[[^\]]*\]\([^)]+\)", text)
    headings = [ln for ln in lines if ln.lstrip().startswith("#")]
    list_items = [ln for ln in lines if re.match(r"\s*[-*+]\s+", ln)]
    code_fences = low.count("```")
    inline_code = len(re.findall(r"`[^`\n]+`", text))

    buzz = _count_any(low, BUZZWORDS)
    wip = _count_any(low, WIP_MARKERS)
    exclaim = text.count("!")
    upper_words = sum(1 for w in words if len(w) > 2 and w.isupper())

    return {
        "readme_chars": n_chars,
        "readme_words": len(words),
        "readme_lines": len(lines),
        "emoji_count": n_emoji,
        "emoji_per_1k_chars": round(1000.0 * n_emoji / max(n_chars, 1), 4),
        "rocket_count": rocket,
        "fire_count": fire,
        "sparkles_count": sparkles,
        "image_count": n_images,
        "badge_count": badge_count,
        "badge_density": round(1000.0 * badge_count / max(n_chars, 1), 4),
        "link_count": len(md_links),
        "heading_count": len(headings),
        "list_item_count": len(list_items),
        "code_fence_blocks": code_fences // 2,
        "inline_code_count": inline_code,
        "buzzword_count": buzz,
        "buzzword_per_1k_words": round(1000.0 * buzz / n_words, 4),
        "wip_marker_count": wip,
        "exclamation_count": exclaim,
        "exclamation_per_1k_words": round(1000.0 * exclaim / n_words, 4),
        "uppercase_word_count": upper_words,
        "has_install_section": int("install" in low),
        "has_usage_section": int("usage" in low or "getting started" in low),
        "has_license_section": int("license" in low),
        "has_contributing": int("contribut" in low),
        "has_tests_mention": int("test" in low),
    }


FEATURE_NAMES = list(extract("").keys())


if __name__ == "__main__":
    import json, sys
    src = sys.stdin.read() if not sys.stdin.isatty() else "# 🚀 blazingly fast!"
    print(json.dumps(extract(src), indent=2))
