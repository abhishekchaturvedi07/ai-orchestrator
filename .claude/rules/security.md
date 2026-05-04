# Security Rules

- All new endpoints must use the `HTTPBearer` dependency.
- JWT decoding must use `algorithms=["HS256"]`.
- Identity must be extracted from the `id` field of the payload.
- Log every successful and failed authentication attempt for audit.
