# Context7 Skill

Retrieve up-to-date documentation for software libraries and frameworks.

## Quick Start

```bash
# Search for a library
curl -s "https://context7.com/api/v2/libs/search?libraryName=react&query=hooks" | jq '.results[0].id'

# Fetch documentation
curl -s "https://context7.com/api/v2/context?libraryId=/websites/react_dev_reference&query=useState&type=txt"
```

## When to Use

- Looking up documentation for any programming library
- Finding code examples for specific APIs
- Verifying correct usage of library functions
- Getting current information about APIs that may have changed since training

## Requirements

- `curl` for HTTP requests
- `jq` for JSON processing (optional but recommended)

## License

MIT
