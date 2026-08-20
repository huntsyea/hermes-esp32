"""Bounded text sent to the Companion display."""

MAX_RESPONSE_PREVIEW_BYTES = 96


def response_preview(text: str) -> str:
    encoded = str(text).encode("utf-8")
    if len(encoded) <= MAX_RESPONSE_PREVIEW_BYTES:
        return str(text)
    prefix = encoded[: MAX_RESPONSE_PREVIEW_BYTES - 3]
    while prefix:
        try:
            return prefix.decode("utf-8") + "..."
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return "..."
