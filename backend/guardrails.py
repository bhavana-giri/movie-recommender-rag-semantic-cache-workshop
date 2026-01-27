"""
Guardrails Module for StreamFlix Help Center
=============================================
Uses SemanticRouter to filter out-of-scope queries that are not related
to StreamFlix support topics. Only one route is used to avoid false positives.
"""
from redisvl.extensions.router import SemanticRouter, Route
from redisvl.utils.vectorize import HFTextVectorizer
from redis import Redis

from .config import REDIS_URL, EMBEDDING_MODEL

# Helpful message for out-of-scope queries
OUT_OF_SCOPE_MESSAGE = """I'm StreamFlix's Help Center assistant, and I can only help with StreamFlix-related questions.

Here are some things I can help you with:
- Account issues (password reset, subscription, profiles)
- Playback problems (buffering, quality, audio sync)
- Content questions (availability, downloads, parental controls)
- Device support (smart TVs, mobile apps, casting)
- Billing inquiries (charges, payment methods, refunds)

Please ask a question about StreamFlix, or visit help.streamflix.com for more options."""

# Define StreamFlix support route with comprehensive reference phrases
STREAMFLIX_ROUTE = Route(
    name="streamflix_support",
    references=[
        # Account topics
        "reset password", "forgot password", "change subscription plan",
        "cancel subscription", "update payment method", "create profile",
        "manage profiles", "two-factor authentication", "sign out of devices",
        "account settings", "login issues", "email change",
        # Playback topics
        "video buffering", "playback quality", "audio sync", "subtitles",
        "video error", "streaming issues", "blurry video", "freezing",
        "captions not working", "audio language", "HD quality", "4K streaming",
        # Content topics
        "movie not available", "show not available", "content region",
        "download offline", "parental controls", "continue watching",
        "watchlist", "recommendations", "new releases", "leaving soon",
        # Device topics
        "supported devices", "cast to TV", "app crash", "chromecast",
        "smart TV app", "roku", "fire stick", "apple tv", "mobile app",
        "browser streaming", "multiple devices",
        # Billing topics
        "billing", "payment failed", "unexpected charge", "refund",
        "subscription cost", "plan pricing", "free trial", "invoice",
        # Technical topics
        "internet speed", "contact support", "error code", "app update",
        "connection issues", "VPN", "network requirements",
    ],
    distance_threshold=0.5,  # Tune based on testing (0-2 scale, lower = stricter)
)


def create_guardrail_router(
    redis_client: Redis,
    vectorizer: HFTextVectorizer
) -> SemanticRouter:
    """
    Create a SemanticRouter for guardrail checks.
    
    Args:
        redis_client: Redis connection
        vectorizer: Text vectorizer for embeddings
        
    Returns:
        Configured SemanticRouter instance
    """
    router = SemanticRouter(
        name="help_center_guardrail",
        routes=[STREAMFLIX_ROUTE],
        vectorizer=vectorizer,
        redis_client=redis_client,
    )
    return router
