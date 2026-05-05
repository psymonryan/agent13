"""
Tools package with decorator-based registration and auto-discovery.

Usage:
    from tools import tool, TOOLS, execute_tool

    @tool
    def my_function(arg: str) -> str:
        '''Description of function.

        Args:
            arg: Description of arg

        Returns:
            Description of return value
        '''
        return arg
"""

import os
import json
import asyncio
import inspect
import importlib
from typing import get_type_hints, get_origin, get_args, Callable, Union

# Lazy import to avoid circular dependency with agent package
_log_event = None
_log_error = None
_truncate_for_log = None


def _get_log_functions():
    global _log_event, _log_error, _truncate_for_log  # noqa: F824
    if _log_event is None:
        try:
            from agent13.debug_log import log_event, log_error, truncate_for_log

            globals()["_log_event"] = log_event
            globals()["_log_error"] = log_error
            globals()["_truncate_for_log"] = truncate_for_log
        except ImportError:
            # Logging not available, use no-op functions
            globals()["_log_event"] = lambda *args, **kwargs: None
            globals()["_log_error"] = lambda *args, **kwargs: None
            globals()["_truncate_for_log"] = lambda x, max_len=100: (
                x[:max_len] if len(x) > max_len else x
            )
    return _log_event, _log_error, _truncate_for_log


# Registry for sync tools
_registry: dict[str, Callable] = {}
# Registry for async tools
_async_registry: dict[str, Callable] = {}
_schemas: list[dict] = []
# Map of tool name -> list of group names
_tool_groups_map: dict[str, list[str]] = {}
_discovered = False


def _python_type_to_json_schema(python_type: type) -> dict:
    """Convert Python type hints to JSON schema types."""
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    # Handle Optional[X] which is Union[X, None]
    origin = get_origin(python_type)
    if origin is Union:
        args = get_args(python_type)
        # Find the non-None type and recurse
        for arg in args:
            if arg is not type(None):
                return _python_type_to_json_schema(arg)

    # Handle generic types like list[str], list[int], etc.
    if origin is list:
        args = get_args(python_type)
        if args:
            item_type = _python_type_to_json_schema(args[0])
            return {"type": "array", "items": item_type}
        return {"type": "array"}

    return type_map.get(python_type, {"type": "string"})


def _parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """
    Parse docstring to extract description and parameter descriptions.

    Returns:
        Tuple of (main_description, param_descriptions dict)
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().split("\n")
    main_desc = []
    param_descs = {}
    current_param = None
    in_args_section = False

    for line in lines:
        stripped = line.strip()

        # Check for Args: section
        if stripped.lower().startswith("args:"):
            in_args_section = True
            continue

        # Check for other sections that end Args:
        if stripped.lower().startswith(("returns:", "raises:", "example:", "note:")):
            in_args_section = False
            current_param = None
            continue

        if in_args_section:
            # Parse parameter line: "param: description"
            if ":" in stripped and not stripped.startswith(" "):
                parts = stripped.split(":", 1)
                current_param = parts[0].strip()
                param_descs[current_param] = parts[1].strip() if len(parts) > 1 else ""
            elif current_param and stripped:
                # Continuation of previous param description
                param_descs[current_param] += " " + stripped
        else:
            # Main description
            if stripped:
                main_desc.append(stripped)

    return " ".join(main_desc), param_descs


def tool(
    func: Callable | None = None,
    *,
    is_async: bool = False,
    timeout: float | None = None,
    groups: list[str] | None = None,
) -> Callable:
    """
    Decorator to register a function as a tool.

    Supports both @tool and @tool(is_async=True) syntax.

    Args:
        func: The tool function (when used as @tool without parens)
        is_async: Whether the function is async (default: False)
        timeout: Maximum execution time in seconds (default: None)
        groups: List of group names this tool belongs to (default: None)
                Tools in the "devel" group are hidden unless --devel is active.

    Extracts schema from type hints and docstring.
    """

    def decorator(f: Callable) -> Callable:
        name = f.__name__
        docstring = f.__doc__ or ""
        description, param_descs = _parse_docstring(docstring)

        # Store timeout in function metadata
        if timeout is not None:
            f._tool_timeout = timeout

        # Store groups in function metadata
        if groups:
            f._tool_groups = groups
        else:
            f._tool_groups = []

        # Get type hints
        hints = get_type_hints(f)
        sig = inspect.signature(f)

        # Build parameters schema
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue

            param_type = hints.get(param_name, str)
            param_schema = _python_type_to_json_schema(param_type)

            # Add description from docstring if available
            if param_name in param_descs:
                param_schema["description"] = param_descs[param_name]

            properties[param_name] = param_schema

            # Mark as required if no default value
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Build the full schema
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or f"Execute {name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }

        if required:
            schema["function"]["parameters"]["required"] = required

        # Register in appropriate registry
        if is_async:
            _async_registry[name] = f
        else:
            _registry[name] = f
        _schemas.append(schema)

        # Track group membership
        if groups:
            _tool_groups_map[name] = groups

        return f

    # Handle both @tool and @tool(...) calling conventions
    if func is not None:
        return decorator(func)
    return decorator


def _get_coercion_type(expected_type) -> type | None:
    """Extract the type to coerce to from a type hint.

    Handles simple types (int, float, bool, str) and Optional types
    (int | None, str | None). Returns None if coercion is not
    needed/supported.
    """
    # Direct match for simple types
    if expected_type in (int, float, bool, str):
        return expected_type

    # Handle Optional[X] which is Union[X, None]
    origin = get_origin(expected_type)
    if origin is Union:
        args = get_args(expected_type)
        # Look for a coercible type in the union args
        for arg in args:
            if arg in (int, float, bool, str):
                return arg

    return None


async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a registered tool by name with given arguments.

    This is async to support both sync and async tools.
    Sync tools are run in a thread pool to avoid blocking.
    """
    # Ensure tools are discovered before execution
    _ensure_discovered()

    log_event, log_error, truncate_for_log = _get_log_functions()

    # Check async registry first
    if name in _async_registry:
        return await _execute_async_tool(name, arguments, log_event, log_error)

    # Check sync registry
    if name not in _registry:
        error_msg = f"Unknown tool: {name}"
        log_event(
            "tool_error", {"name": name, "error": error_msg, "arguments": arguments}
        )
        return json.dumps({"error": error_msg})

    # Sync tool - run in executor to avoid blocking
    return await _execute_sync_tool_in_executor(name, arguments, log_event, log_error)


def _coerce_arguments(func: Callable, arguments: dict) -> dict:
    """Coerce argument types based on function signature.

    Handles forward coercion (str→int/float/bool) and reverse coercion
    (int/float/bool→str) as well as float→int and bool→int conversions.
    Uncoerceable types (e.g. list→str) produce clear errors.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    coerced_args = {}

    for param_name, param in sig.parameters.items():
        if param_name not in arguments:
            continue

        value = arguments[param_name]
        expected_type = hints.get(param_name)

        # Determine if we need to coerce and to what type
        coerce_to = _get_coercion_type(expected_type)

        # Already the correct type — no coercion needed
        # Use type() not isinstance() to avoid bool/isinstance(int) trap
        if coerce_to is not None and type(value) is coerce_to:
            coerced_args[param_name] = value
            continue

        # --- Forward coercion: str → int/float/bool ---
        if isinstance(value, str) and coerce_to is not None:
            try:
                if coerce_to is int:
                    value = int(value)
                elif coerce_to is float:
                    value = float(value)
                elif coerce_to is bool:
                    # Handle common string representations of booleans
                    if value.lower() in ("true", "1", "yes"):
                        value = True
                    elif value.lower() in ("false", "0", "no"):
                        value = False
                    else:
                        value = bool(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Parameter '{param_name}' expects {coerce_to.__name__}, "
                    f"got string '{value}' that cannot be converted"
                )

        # --- Reverse coercion: non-str → str ---
        elif coerce_to is str and not isinstance(value, str):
            if isinstance(value, bool):
                # JSON booleans — convert to "true"/"false"
                value = "true" if value else "false"
            elif isinstance(value, (int, float)):
                value = str(value)
            else:
                # list, dict, None, etc — reject clearly
                raise ValueError(
                    f"Parameter '{param_name}' expects str, "
                    f"got {type(value).__name__} which cannot be converted"
                )

        # --- Float → Int coercion ---
        elif coerce_to is int and isinstance(value, float):
            # Reject non-integer floats (e.g. 8.5 → error)
            if value != int(value):
                raise ValueError(
                    f"Parameter '{param_name}' expects int, "
                    f"got {value} which is not a whole number"
                )
            value = int(value)

        # --- Bool → Int coercion ---
        elif coerce_to is int and isinstance(value, bool):
            value = 1 if value else 0

        coerced_args[param_name] = value

    return coerced_args


async def _execute_async_tool(name: str, arguments: dict, log_event, log_error) -> str:
    """Execute an async tool."""
    try:
        func = _async_registry[name]
        coerced_args = _coerce_arguments(func, arguments)

        # Check for timeout
        timeout = getattr(func, "_tool_timeout", None)
        if timeout:
            result = await asyncio.wait_for(func(**coerced_args), timeout=timeout)
        else:
            result = await func(**coerced_args)

        result_str = (
            json.dumps(result, indent=2)
            if isinstance(result, (dict, list))
            else json.dumps(result)
        )
        log_event(
            "tool_execution",
            {
                "name": name,
                "arguments": arguments,
                "result": result_str,
            },
        )
        return result_str
    except asyncio.TimeoutError:
        error_msg = f"Tool timed out after {timeout} seconds"
        log_error(
            TimeoutError(error_msg),
            {"context": "tool_execution", "name": name, "arguments": arguments},
        )
        return json.dumps({"error": error_msg})
    except Exception as e:
        log_error(
            e, {"context": "tool_execution", "name": name, "arguments": arguments}
        )
        return json.dumps({"error": str(e)})


async def _execute_sync_tool_in_executor(
    name: str, arguments: dict, log_event, log_error
) -> str:
    """Execute a sync tool in a thread pool executor."""
    try:
        func = _registry[name]
        coerced_args = _coerce_arguments(func, arguments)

        # Run sync function in executor
        loop = asyncio.get_running_loop()

        # Check for timeout
        timeout = getattr(func, "_tool_timeout", None)
        if timeout:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(**coerced_args)),
                timeout=timeout,
            )
        else:
            result = await loop.run_in_executor(None, lambda: func(**coerced_args))

        result_str = (
            json.dumps(result, indent=2)
            if isinstance(result, (dict, list))
            else json.dumps(result)
        )
        log_event(
            "tool_execution",
            {
                "name": name,
                "arguments": arguments,
                "result": result_str,
            },
        )
        return result_str
    except asyncio.TimeoutError:
        error_msg = f"Tool timed out after {timeout} seconds"
        log_error(
            TimeoutError(error_msg),
            {"context": "tool_execution", "name": name, "arguments": arguments},
        )
        return json.dumps({"error": error_msg})
    except Exception as e:
        log_error(
            e, {"context": "tool_execution", "name": name, "arguments": arguments}
        )
        return json.dumps({"error": str(e)})


def get_tools() -> list[dict]:
    """Get list of all registered tool schemas."""
    _ensure_discovered()
    return _schemas.copy()


def name_matches(name: str, patterns: list[str]) -> bool:
    """Check if a name matches any of the provided patterns.

    Supports two forms (case-insensitive):
    - Glob wildcards using fnmatch (e.g., 'tui_*')
    - Regex when prefixed with 're:' (e.g., 're:^tui.*')

    Args:
        name: Tool name to check
        patterns: List of patterns to match against

    Returns:
        True if name matches any pattern
    """
    import re as _re
    import functools
    from fnmatch import fnmatch

    @functools.lru_cache(maxsize=256)
    def _compile_icase(expr: str):
        try:
            return _re.compile(expr, _re.IGNORECASE)
        except _re.error:
            return None

    n = name.lower()
    for raw in patterns:
        if not (p := (raw or "").strip()):
            continue

        if p.startswith("re:"):
            rx = _compile_icase(p.removeprefix("re:"))
            if rx is not None and rx.fullmatch(name) is not None:
                return True
        elif fnmatch(n, p.lower()):
            return True

    return False


def get_filtered_tools(
    devel: bool = False,
    skills: bool = False,
    enabled_tools: list[str] | None = None,
    disabled_tools: list[str] | None = None,
) -> list[dict]:
    """Get tool schemas filtered by devel/skills mode and config-level allow/deny lists.

    Filtering is applied in order:
    1. Group filter: tools in the "devel" group are hidden unless devel=True
    2. Group filter: tools in the "skills" group are hidden unless skills=True
    3. Config filter: if enabled_tools is non-empty, only matching tools pass
       (whitelist); otherwise disabled_tools acts as a blacklist.

    Args:
        devel: If True, include tools in the "devel" group (default: False)
        skills: If True, include tools in the "skills" group (default: False)
        enabled_tools: Whitelist patterns (empty/None = all pass)
        disabled_tools: Blacklist patterns (applied only if enabled_tools empty)

    Returns:
        Filtered list of tool schemas
    """
    _ensure_discovered()

    # Start with all schemas
    result = []
    for schema in _schemas:
        tool_name = schema["function"]["name"]

        # 1. Group filter: hide "devel" group unless in devel mode
        tool_groups = _tool_groups_map.get(tool_name, [])
        if not devel and "devel" in tool_groups:
            continue

        # 2. Group filter: hide "skills" group unless in skills mode
        if not skills and "skills" in tool_groups:
            continue

        # 3. Config-level enabled/disabled filter
        if enabled_tools:
            # Whitelist mode: only tools matching enabled_tools pass
            if not name_matches(tool_name, enabled_tools):
                continue
        elif disabled_tools:
            # Blacklist mode: tools matching disabled_tools are excluded
            if name_matches(tool_name, disabled_tools):
                continue

        result.append(schema)

    return result


def get_tool_groups(name: str) -> list[str]:
    """Get the groups a tool belongs to.

    Args:
        name: Tool name

    Returns:
        List of group names (empty if tool has no groups)
    """
    _ensure_discovered()
    return _tool_groups_map.get(name, [])


def get_tool_names() -> list[str]:
    """Get list of all registered tool names (sync and async)."""
    _ensure_discovered()
    return list(_registry.keys()) + list(_async_registry.keys())


def is_tool_async(name: str) -> bool:
    """Check if a tool is async."""
    _ensure_discovered()
    return name in _async_registry


def get_async_tools() -> list[str]:
    """Get list of async tool names."""
    _ensure_discovered()
    return list(_async_registry.keys())


def get_sync_tools() -> list[str]:
    """Get list of sync tool names."""
    _ensure_discovered()
    return list(_registry.keys())


def _ensure_discovered():
    """Lazily discover tools on first access."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    _auto_discover()


def _auto_discover():
    """Auto-discover and import all tool modules in this package."""
    package_dir = os.path.dirname(__file__)

    for filename in os.listdir(package_dir):
        if (
            filename.endswith(".py")
            and filename != "__init__.py"
            and not filename.startswith("_")
        ):
            module_name = filename[:-3]  # Remove .py extension
            try:
                importlib.import_module(f".{module_name}", package=__name__)
            except Exception as e:
                print(
                    f"Warning: Failed to import tool module {module_name}: {e}",
                    file=__import__("sys").stderr,
                )


# Export TOOLS for backward compatibility
TOOLS = _schemas
