def run_pre_routing_hook(query: str) -> str | None:
    """
    Layer 3 Guardrail: Scans the input for malicious intents or restricted operations
    before it ever reaches the LLM.
    Returns an error message if blocked, or None if safe.
    """
    print("🛡️ [GUARDRAIL] Running Pre-Routing Security Hook...")
    query_lower = query.lower()

    # Define enterprise blacklisted phrases
    blocked_patterns = [
        "ignore all previous instructions",
        "system prompt",
        "drop table",
        "rm -rf",
        "aws_access_key_id"
    ]

    for pattern in blocked_patterns:
        if pattern in query_lower:
            print(f"🚨 [SECURITY VIOLATION] Blocked pattern detected: '{pattern}'")
            return "SECURITY_ALERT: Your query violates Enterprise DevSecOps policies and has been blocked."

    return None # The query is safe!