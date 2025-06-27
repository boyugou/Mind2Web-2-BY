import importlib.util
from pathlib import Path
import sys

def get_formatted_tool_sources(package_name="mind2web2", tools_subdir="api_tools") -> str:
    """
    Retrieve formatted paths and source codes of all *.py files (excluding __init__.py)
    located within a specified subdirectory of a package.

    Args:
        package_name (str): Name of the package containing the tools. Default is "mind2web2".
        tools_subdir (str): Subdirectory within the package to search. Default is "api_tools".

    Returns:
        str: Formatted string containing paths and source code of tools.
    """
    spec = importlib.util.find_spec(f"{package_name}.{tools_subdir}")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError(f"❌ Could not find installed module {package_name}.{tools_subdir}. Check package installation and environment.")

    tools_dir = Path(spec.submodule_search_locations[0])
    package_dir = tools_dir.parent

    result = []
    for py_path in sorted(tools_dir.rglob("*.py")):
        if py_path.name == "__init__.py":
            continue

        rel_path = py_path.relative_to(package_dir.parent)
        tool_source = py_path.read_text(encoding="utf-8")

        formatted_tool = (
            f"{'#' * 88}\n"
            f"TOOL PATH: {rel_path.as_posix()}\n"
            f"{'#' * 88}\n"
            f"{tool_source}\n"
        )

        result.append(formatted_tool)

    return "\n".join(result)

# Example usage
if __name__ == "__main__":
    formatted_sources = get_formatted_tool_sources()
    print(formatted_sources)