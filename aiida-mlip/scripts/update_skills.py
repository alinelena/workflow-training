#!/usr/bin/env python3
"""
Update script for AiiDA-MLIP skills and reference documentation.

Fetches the latest updates from:
1. GitHub Repository: https://github.com/stfc/aiida-mlip
2. Documentation:     https://stfc.github.io/aiida-mlip/

Parses AiiDA entry points, calculation input/output specs, parsers, and example
submission scripts, keeping the skill references in sync.

Usage:
    python update_skills.py [--repo <path-or-url>] [--docs-url <url>] [--check] [--force] [--verbose]
"""

import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_REPO_URL = "https://github.com/stfc/aiida-mlip.git"
DEFAULT_DOCS_URL = "https://stfc.github.io/aiida-mlip/"


def log(msg: str, verbose: bool = False, is_verbose_only: bool = False) -> None:
    if is_verbose_only and not verbose:
        return
    print(f"[aiida-mlip-updater] {msg}")


def check_docs_live(docs_url: str, verbose: bool = False) -> dict:
    """Check if the documentation website is reachable and extract metadata."""
    log(f"Checking documentation URL: {docs_url} ...", verbose)
    result = {"url": docs_url, "accessible": False, "version": None}
    try:
        req = urllib.request.Request(
            docs_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AiiDAMLIPSkillsUpdater/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result["status"] = response.status
            if response.status == 200:
                result["accessible"] = True
                content = response.read(65536).decode("utf-8", errors="ignore")
                match = re.search(r"aiida-mlip\s+v?([0-9]+\.[0-9]+(?:\.[0-9]+)?)", content, re.IGNORECASE)
                if match:
                    result["version"] = match.group(1)
                log(f"Documentation reachable (HTTP 200). Detected version: {result['version'] or 'unknown'}", verbose)
    except Exception as e:
        log(f"Warning: Could not fetch documentation URL '{docs_url}': {e}", verbose)
    return result


def resolve_repo(repo_input: str, verbose: bool = False) -> Path:
    """Resolve repository path (clone if remote URL, use directly if local directory)."""
    p = Path(repo_input)
    if p.exists() and (p / "pyproject.toml").exists():
        log(f"Using local repository clone at: {p.resolve()}", verbose)
        return p.resolve()

    cache_dir = Path("/tmp/aiida-mlip-upstream")
    if cache_dir.exists() and (cache_dir / ".git").exists():
        log(f"Updating existing upstream clone in {cache_dir}...", verbose)
        subprocess.run(["git", "-C", str(cache_dir), "fetch", "--depth=1", "origin", "main"], check=False)
        subprocess.run(["git", "-C", str(cache_dir), "reset", "--hard", "origin/main"], check=False)
    else:
        log(f"Cloning upstream repository from {repo_input} into {cache_dir}...", verbose)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_input, str(cache_dir)],
            check=True,
        )
    return cache_dir


def extract_version_from_pyproject(repo_path: Path) -> str:
    """Extract version from pyproject.toml."""
    pyproject = repo_path / "pyproject.toml"
    if not pyproject.exists():
        return "unknown"
    for line in pyproject.read_text().splitlines():
        m = re.match(r'^\s*version\s*=\s*["\']([^"\']+)["\']', line)
        if m:
            return m.group(1)
    return "unknown"


def parse_entry_points(repo_path: Path) -> dict:
    """Parse entry points from pyproject.toml."""
    pyproject = repo_path / "pyproject.toml"
    res = {
        "calculations": {},
        "data": {},
        "parsers": {},
    }
    if not pyproject.exists():
        return res

    current_section = None
    for line in pyproject.read_text().splitlines():
        line = line.strip()
        if line.startswith("[project.entry-points."):
            m = re.search(r'\[project\.entry-points\."aiida\.(\w+)"\]', line)
            if m:
                current_section = m.group(1)
            else:
                current_section = None
            continue
        if line.startswith("["):
            current_section = None
            continue

        if current_section and "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            key = key.strip(' "')
            val = val.strip(' "')
            if current_section in res:
                res[current_section][key] = val

    return res


def parse_calc_ports(repo_path: Path) -> dict:
    """Parse define method in calculation classes to extract input ports."""
    calc_dir = repo_path / "aiida_mlip" / "calculations"
    ports_by_calc = {}
    if not calc_dir.exists():
        return ports_by_calc

    for py_file in calc_dir.glob("*.py"):
        if py_file.name in ("__init__.py", "base.py"):
            continue
        calc_name = py_file.stem
        ports = []
        try:
            content = py_file.read_text()
            # Regex match spec.input lines
            for line in content.splitlines():
                m = re.search(r'spec\.input\(\s*["\']([\w.]+)["\']\s*,\s*valid_type=([^,\)]+)', line)
                if m:
                    port_name = m.group(1)
                    port_type = m.group(2).strip()
                    req = "required=False" not in line
                    ports.append({"name": port_name, "type": port_type, "required": req})
            ports_by_calc[calc_name] = ports
        except Exception as e:
            log(f"Warning: Failed to parse {py_file.name}: {e}")

    return ports_by_calc


def generate_entry_points_ref(entry_points: dict, out_path: Path) -> None:
    """Generate references/entry_points.md."""
    lines = [
        "# AiiDA-MLIP Registered Entry Points",
        "",
        "Registered entry points exposed by `aiida-mlip` in the AiiDA registry:",
        "",
        "## Calculations (`aiida.calculations`)",
        "",
        "| Entry Point | Factory Call | Implementation Class |",
        "| :--- | :--- | :--- |",
    ]
    for ep, cls in sorted(entry_points.get("calculations", {}).items()):
        lines.append(f"| `{ep}` | `CalculationFactory('{ep}')` | `{cls}` |")

    lines.extend([
        "",
        "## Data Types (`aiida.data`)",
        "",
        "| Entry Point | Factory Call | Implementation Class |",
        "| :--- | :--- | :--- |",
    ])
    for ep, cls in sorted(entry_points.get("data", {}).items()):
        lines.append(f"| `{ep}` | `DataFactory('{ep}')` | `{cls}` |")

    lines.extend([
        "",
        "## Parsers (`aiida.parsers`)",
        "",
        "| Entry Point | Implementation Class |",
        "| :--- | :--- |",
    ])
    for ep, cls in sorted(entry_points.get("parsers", {}).items()):
        lines.append(f"| `{ep}` | `{cls}` |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def generate_calc_overview_ref(entry_points: dict, out_path: Path) -> None:
    """Generate references/calculations_overview.md."""
    lines = [
        "# AiiDA-MLIP Calculations Overview",
        "",
        "`aiida-mlip` executes `janus-core` calculations within the AiiDA automated provenance workflow system.",
        "",
        "## Supported Calculation Types",
        "",
        "| Calculation | Entry Point | Purpose |",
        "| :--- | :--- | :--- |",
        "| Single Point | `mlip.sp` | Static energy, forces, stress, Hessians |",
        "| Geometry Optimization | `mlip.opt` | Force minimization & cell relaxation |",
        "| Molecular Dynamics | `mlip.md` | NVE, NVT, NPT finite temperature dynamics |",
        "| Phonons | `mlip.ph` | Phonon band structures, DOS, and thermodynamics |",
        "| Descriptors | `mlip.descriptors` | MLIP invariant representations extraction |",
        "| Train / Fine-Tune | `mlip.train` | MACE and MLIP model training and transfer learning |",
        "",
        "## Common Input Ports Across All Calculations",
        "",
        "- `code` (`InstalledCode`): The registered `janus` executable (e.g. `janus@localhost`).",
        "- `metadata.options.resources`: Execution machine resources (e.g. `{'num_machines': 1}`).",
        "- `structure` (`StructureData` or `XyzData`): The target atomic configuration.",
        "- `architecture` (`Str`): MLIP architecture (`mace_mp`, `sevennet`, `chgnet`, etc.).",
        "- `model` (`ModelData`): Checkpoint file or URI to trained model weights.",
        "- `config` (`JanusConfigfile`): Optional YAML configuration node supplying calculation arguments.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def generate_ports_ref(calc_name: str, ports: list, out_path: Path) -> None:
    """Generate references/ports.md for an individual calculation."""
    lines = [
        f"# Input and Output Ports Reference for {calc_name.title()}",
        "",
        "| Port | Valid Type | Required | Description |",
        "| :--- | :--- | :---: | :--- |",
    ]
    for p in ports:
        req_str = "Yes" if p["required"] else "No"
        lines.append(f"| `{p['name']}` | `{p['type']}` | {req_str} | Input argument for `{calc_name}` |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def sync_examples(repo_path: Path, output_dir: Path, verbose: bool = False) -> None:
    """Synchronize submission scripts from upstream examples into skills."""
    calc_examples = repo_path / "examples" / "calculations"
    wf_examples = repo_path / "examples" / "workflows"
    skills_root = output_dir / "skills"

    mapping = {
        calc_examples / "submit_singlepoint.py": skills_root / "aiida-mlip-singlepoint" / "examples" / "submit_singlepoint.py",
        calc_examples / "submit_geomopt.py": skills_root / "aiida-mlip-geomopt" / "examples" / "submit_geomopt.py",
        calc_examples / "submit_md.py": skills_root / "aiida-mlip-md" / "examples" / "submit_md.py",
        calc_examples / "submit_md_using_config.py": skills_root / "aiida-mlip-md" / "examples" / "submit_md_using_config.py",
        calc_examples / "submit_phonon.py": skills_root / "aiida-mlip-phonons" / "examples" / "submit_phonon.py",
        calc_examples / "submit_descriptors.py": skills_root / "aiida-mlip-descriptors" / "examples" / "submit_descriptors.py",
        calc_examples / "submit_train.py": skills_root / "aiida-mlip-train" / "examples" / "submit_train.py",
        wf_examples / "submit_ht_workgraph.py": skills_root / "aiida-mlip-workflows" / "examples" / "submit_ht_workgraph.py",
    }

    copied = 0
    for src_path, dst_path in mapping.items():
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied += 1
            log(f"Copied example: {src_path.name} -> {dst_path.relative_to(output_dir)}", verbose, is_verbose_only=True)

    log(f"Synchronized {copied} example submission scripts from upstream examples.", verbose)


def update_version_and_plugin(version: str, output_dir: Path, verbose: bool = False) -> None:
    """Update VERSION and plugin.json files."""
    version_file = output_dir / "VERSION"
    version_file.write_text(f"{version}\n")

    plugin_json = output_dir / "plugin.json"
    data = {}
    if plugin_json.exists():
        try:
            data = json.loads(plugin_json.read_text())
        except Exception:
            data = {}
    data["name"] = "aiida-mlip"
    data["version"] = version
    data["description"] = (
        "Comprehensive AI skills, execution rules, and workflow automation for STFC aiida-mlip "
        "(machine learned interatomic potentials AiiDA plugin)."
    )
    data["homepage"] = DEFAULT_DOCS_URL
    data["repository"] = {"type": "git", "url": DEFAULT_REPO_URL}
    data["author"] = {"name": "STFC / AiiDA-MLIP Contributors", "url": "https://github.com/stfc/aiida-mlip"}
    plugin_json.write_text(json.dumps(data, indent=2) + "\n")
    log(f"Updated {version_file.name} and {plugin_json.name} with version {version}.", verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Update AiiDA-MLIP AI skills and references using upstream GitHub repo and documentation."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_URL,
        help=f"Path to local aiida-mlip clone or Git repository URL (default: {DEFAULT_REPO_URL})",
    )
    parser.add_argument(
        "--docs-url",
        default=DEFAULT_DOCS_URL,
        help=f"Published documentation website URL (default: {DEFAULT_DOCS_URL})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root directory of the aiida-mlip skill group (default: auto-detected)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check upstream for updates and differences without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of reference files even if versions match",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose diagnostics",
    )

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path(__file__).resolve().parent.parent

    log(f"Target aiida-mlip skill group directory: {output_dir}")

    # 1. Check documentation website
    docs_info = check_docs_live(args.docs_url, verbose=args.verbose)

    # 2. Resolve repository
    repo_path = resolve_repo(args.repo, verbose=args.verbose)
    upstream_version = extract_version_from_pyproject(repo_path)
    log(f"Upstream repository version (pyproject.toml): {upstream_version}")

    # 3. Read current local version
    local_version_file = output_dir / "VERSION"
    local_version = local_version_file.read_text().strip() if local_version_file.exists() else "none"
    log(f"Local installed version: {local_version}")

    # 4. Parse upstream entry points and calculation ports
    entry_points = parse_entry_points(repo_path)
    calc_ports = parse_calc_ports(repo_path)
    log(f"Parsed {len(entry_points.get('calculations', {}))} calculations and {len(entry_points.get('data', {}))} data types.")

    if args.check:
        log("=== Check Mode Summary ===")
        log(f"Local Version:    {local_version}")
        log(f"Upstream Version: {upstream_version}")
        log(f"Docs Live:        {docs_info['accessible']} (version: {docs_info['version']})")
        log(f"Calculations:     {', '.join(sorted(entry_points.get('calculations', {}).keys()))}")
        if local_version != upstream_version:
            log(f"Update available: {local_version} -> {upstream_version}. Run without --check to apply.")
        else:
            log("Skills are currently up-to-date with upstream.")
        return

    # 5. Apply updates
    log(f"Updating reference documentation and examples for v{upstream_version}...")

    skills_root = output_dir / "skills"
    core_ref_dir = skills_root / "aiida-mlip-core" / "references"
    generate_entry_points_ref(entry_points, core_ref_dir / "entry_points.md")
    generate_calc_overview_ref(entry_points, core_ref_dir / "calculations_overview.md")

    # Generate port references for each calculation skill
    skill_calc_map = {
        "singlepoint": skills_root / "aiida-mlip-singlepoint" / "references" / "ports.md",
        "geomopt": skills_root / "aiida-mlip-geomopt" / "references" / "ports.md",
        "md": skills_root / "aiida-mlip-md" / "references" / "ports.md",
        "phonon": skills_root / "aiida-mlip-phonons" / "references" / "ports.md",
        "descriptors": skills_root / "aiida-mlip-descriptors" / "references" / "ports.md",
        "train": skills_root / "aiida-mlip-train" / "references" / "ports.md",
    }
    for calc_stem, port_file in skill_calc_map.items():
        ports = calc_ports.get(calc_stem, [])
        if ports:
            generate_ports_ref(calc_stem, ports, port_file)

    # Sync examples
    sync_examples(repo_path, output_dir, verbose=args.verbose)

    # Update VERSION and plugin.json
    update_version_and_plugin(upstream_version, output_dir, verbose=args.verbose)

    log("✓ Successfully updated AiiDA-MLIP skills, reference files, and examples.")


if __name__ == "__main__":
    main()

