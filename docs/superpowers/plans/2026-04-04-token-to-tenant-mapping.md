# Token-to-Tenant Mapping — Architecture Decision

> **Status:** Superseded — tenant identity resolution will be handled by a centralized gateway (`mcp-oauth-server`), not per-MCP token mapping.

## Decision

Each downstream MCP server (writing-library, cerebellum, zotero, etc.) will **not** implement its own token-to-client_id mapping. Instead:

1. A centralized **`mcp-oauth-server`** gateway fronts all Railway MCP servers
2. The gateway validates Bearer tokens and resolves them to a `client_id`
3. The gateway passes `client_id` to downstream MCP servers via MCP context negotiation
4. Each MCP server reads `ctx.client_id` via `_user_id(ctx)` — **no changes needed**

## Why centralized

- Single source of truth for tenant identity — no duplicating token maps across N servers
- Scales to any number of MCP servers without per-server config
- Future-proof: can swap static tokens for JWT/OAuth without touching downstream servers
- Downstream servers stay simple — they trust the gateway context

## What the writing-library needs (nothing)

The writing-library already reads `ctx.client_id` in `_user_id(ctx)` at `src/server.py:39-44`. Once the gateway sets `client_id` correctly, per-user collection isolation works automatically:

```
Gateway (mcp-oauth-server)
  → validates token → resolves client_id
    → passes client_id via MCP context negotiation
      → writing-library: _user_id(ctx) → ctx.client_id → "acme-corp"
        → collections: acme-corp_writing_passages, acme-corp_writing_terms, ...
```

## What needs to be built (in mcp-oauth-server, not here)

See: `/Users/danilodasilva/Documents/Programming/mcp-servers/mcp-oauth-server/` (to be created)

1. Token validation (static map initially, JWT later)
2. client_id resolution from token
3. MCP context negotiation to pass client_id downstream
4. Proxy/routing to downstream MCP servers on Railway

## Current workaround

Until the gateway exists, the writing-library operates in two modes:
- **stdio**: single-tenant, all data under `default_*` collections
- **HTTP**: `BearerAuthMiddleware` validates access but all tenants collapse to `"default"` — functional but not isolated

This is acceptable for single-user deployment. Multi-tenant isolation requires the gateway.
