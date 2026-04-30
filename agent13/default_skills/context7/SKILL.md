---
name: context7
version: "1.0.0"
description: |
  Retrieve up-to-date documentation for software libraries and frameworks via the Context7 API.
  
  What: Queries the Context7 API to fetch current library documentation, code examples, and API references.
  When: Use when looking up documentation for any programming library, finding code examples for specific APIs, verifying correct usage of library functions, or getting current information about APIs that may have changed since training.
author: Agent13 manage-skills skill
license: MIT
compatibility: Requires curl and jq for JSON processing
metadata:
  created: 2025-01-09
  updated: 2025-01-09
  category: documentation
  tags:
    - documentation
    - context
    - libraries
    - api-reference
---

# Context7

## Overview

This skill enables retrieval of current documentation for software libraries by querying the Context7 API. Use it instead of relying on potentially outdated training data.

## Workflow

### Step 1: Search for the Library

Find the Context7 library ID:

```bash
curl -s "https://context7.com/api/v2/libs/search?libraryName=LIBRARY_NAME&query=TOPIC" | jq '.results[0]'
```

**Parameters:**
- `libraryName` (required): Library name (e.g., "react", "nextjs", "fastapi", "axios")
- `query` (required): Topic description for relevance ranking

**Response fields:**
- `id`: Library identifier for context endpoint
- `title`: Human-readable library name
- `description`: Brief description
- `totalSnippets`: Number of available snippets

### Step 2: Fetch Documentation

Retrieve documentation using the library ID:

```bash
curl -s "https://context7.com/api/v2/context?libraryId=LIBRARY_ID&query=TOPIC&type=txt"
```

**Parameters:**
- `libraryId` (required): Library ID from search results
- `query` (required): Specific topic to retrieve
- `type` (optional): `json` (default) or `txt` (more readable)

## Examples

### React hooks

```bash
# Find React library ID
curl -s "https://context7.com/api/v2/libs/search?libraryName=react&query=hooks" | jq '.results[0].id'

# Fetch useState documentation
curl -s "https://context7.com/api/v2/context?libraryId=/websites/react_dev_reference&query=useState&type=txt"
```

### Next.js routing

```bash
# Find Next.js library ID
curl -s "https://context7.com/api/v2/libs/search?libraryName=nextjs&query=routing" | jq '.results[0].id'

# Fetch app router documentation
curl -s "https://context7.com/api/v2/context?libraryId=/vercel/next.js&query=app+router&type=txt"
```

### FastAPI dependency injection

```bash
# Find FastAPI library ID
curl -s "https://context7.com/api/v2/libs/search?libraryName=fastapi&query=dependencies" | jq '.results[0].id'

# Fetch dependency injection documentation
curl -s "https://context7.com/api/v2/context?libraryId=/fastapi/fastapi&query=dependency+injection&type=txt"
```

## Tips

- Use `type=txt` for readable output
- Use `jq` to filter JSON responses
- Be specific with `query` for better relevance
- Check additional results if first is incorrect
- URL-encode spaces with `+` or `%20`
- No API key required for basic usage (rate-limited)
