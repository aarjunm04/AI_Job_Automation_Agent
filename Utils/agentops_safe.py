"""Absolute Kill Switch for AgentOps."""

def safe_track_agent(name: str = ""):
    def decorator(cls):
        return cls
    return decorator

def safe_track_tool(fn):
    return fn
