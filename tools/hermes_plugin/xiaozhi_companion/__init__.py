def register(ctx):
    """Load the Hermes-dependent adapter only when Hermes activates the plugin."""
    from .adapter import register as register_adapter

    return register_adapter(ctx)

__all__ = ["register"]
