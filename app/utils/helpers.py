"""
Utility functions for Aria pipeline.
"""

def generate_session_id() -> str:
    import uuid
    return str(uuid.uuid4())
