#!/usr/bin/env python3
"""
Update script for Janus-Core skills and reference documentation.

Fetches the latest updates from:
1. GitHub Repository: https://github.com/stfc/janus-core
2. Documentation:     https://stfc.github.io/janus-core/

Parses Typer CLI commands, supported architectures, device types, calculation
options, and example configurations, keeping the skill references in sync.

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

DEFAULT_REPO_URL = "https://github.com/stfc/janus-core.git"
DEFAULT_DOCS_URL = "https://stfc.github.io/janus-core/"


def log(msg: str, verbose: bool = False, is_verbose_only: bool = False) -> None:
    if is_verbose_only and not verbose:
        return
    print(f"[janus-skills-updater] {msg}")


def check_docs_live(docs_url: str, verbose: bool = False) -> dict:
    """Check if the documentation website is reachable and extract metadata."""
    log(f"Checking documentation URL: {docs_url} ...", verbose)
    result = {"url": docs_url, "accessible": False, "version": None}
    try:
        req = urllib.request.Request(
            docs_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JanusSkillsUpdater/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result["status"] = response.status
            if response.status == 200:
                result["accessible"] = True
                content = response.read(65536).decode("utf-8", errors="ignore")
                match = re.search(r"janus-core\s+v?([0-9]+\.[0-9]+(?:\.[0-9]+)?)", content, re.IGNORECASE)
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

    # Remote git URL or directory needing clone
    cache_dir = Path("/tmp/janus-core-upstream")
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


def parse_types(repo_path: Path) -> dict:
    """Parse janus_types.py to extract architectures, devices, ensembles, etc."""
    types_file = repo_path / "janus_core" / "helpers" / "janus_types.py"
    res = {
        "architectures": [],
        "devices": ["cpu", "cuda", "mps", "xpu"],
        "ensembles": [],
        "properties": ["energy", "forces", "stress", "hessian"],
        "eos_names": [],
    }
    if not types_file.exists():
        return res

    content = types_file.read_text()
    try:
        tree = ast.parse(content)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        val_node = node.value
                        if isinstance(val_node, ast.Subscript) and isinstance(val_node.value, ast.Name):
                            if val_node.value.id == "Literal":
                                items = []
                                slice_val = val_node.slice
                                if isinstance(slice_val, ast.Tuple):
                                    elts = slice_val.elts
                                elif isinstance(slice_val, ast.Constant):
                                    elts = [slice_val]
                                else:
                                    elts = []
                                for elt in elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        items.append(elt.value)
                                if name == "Architectures":
                                    res["architectures"] = items
                                elif name == "Devices":
                                    res["devices"] = items
                                elif name == "Ensembles":
                                    res["ensembles"] = items
                                elif name == "Properties":
                                    res["properties"] = items
                                elif name == "EoSNames":
                                    res["eos_names"] = items
    except Exception as e:
        log(f"Warning: Failed to parse AST in janus_types.py: {e}")

    # Fallback regex if AST missed any
    if not res["architectures"]:
        arch_match = re.search(r"Architectures\s*=\s*Literal\[([^\]]+)\]", content)
        if arch_match:
            res["architectures"] = [
                s.strip(" '\"\n\t") for s in arch_match.group(1).split(",") if s.strip(" '\"\n\t")
            ]

    return res


def parse_cli_commands(repo_path: Path) -> dict:
    """Discover registered CLI subcommands and extract their parameters."""
    cli_dir = repo_path / "janus_core" / "cli"
    commands = {}
    if not cli_dir.exists():
        return commands

    # Discover commands registered in janus.py
    janus_py = cli_dir / "janus.py"
    reg_commands = []
    if janus_py.exists():
        for line in janus_py.read_text().splitlines():
            m = re.search(r"\)\((\w+)\)", line)
            if m:
                reg_commands.append(m.group(1))

    if not reg_commands:
        reg_commands = [
            "singlepoint",
            "geomopt",
            "md",
            "phonons",
            "eos",
            "elasticity",
            "neb",
            "descriptors",
            "train",
            "preprocess",
        ]

    for cmd_name in reg_commands:
        cmd_file = cli_dir / f"{cmd_name}.py"
        if not cmd_file.exists():
            continue
        try:
            content = cmd_file.read_text()
            tree = ast.parse(content)
            params = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == cmd_name:
                    for arg in node.args.args:
                        if arg.arg == "ctx":
                            continue
                        params.append(arg.arg)
            commands[cmd_name] = {"file": cmd_file.name, "params": params}
        except Exception as e:
            commands[cmd_name] = {"file": cmd_file.name, "params": []}

    return commands


def generate_architectures_ref(types_info: dict, out_path: Path) -> None:
    """Generate references/architectures.md."""
    archs = types_info.get("architectures", [])
    arch_details = {
        "mace": ("MACE foundation potential", "Molecules and materials"),
        "mace_mp": ("MACE Materials Project potential", "Crystalline and inorganic solids"),
        "mace_off": ("MACE Open Force Field potential", "Organic molecules and solution chemistry"),
        "mace_omol": ("MACE Open Molecule foundation model", "Molecular conformations and reactions"),
        "mace_polar": ("MACE Polarizable potential", "Dielectric and polarization observables"),
        "sevennet": ("SevenNet graph neural network potential", "Inorganic materials, high accuracy"),
        "chgnet": ("Crystal Hamiltonian Graph Neural Network", "Minerals, oxides, battery materials"),
        "nequip": ("E(3)-equivariant interatomic potential", "Niche high-precision systems"),
        "deepmd": ("Deep Potential architecture (DeepMD-kit)", "Condensed matter at scale"),
        "orb": ("Orbital Materials foundation model", "Universal materials potential"),
        "mattersim": ("Microsoft MatterSim foundation potential", "Universal materials simulations"),
        "grace": ("GRACE multi-element potential", "Alloys, intermetallics, crystals"),
        "upet": ("Universal Pretrained Equivariant Transformer", "Broad chemical space"),
        "fairchem": ("FAIRChem / Open Catalyst models", "Surface reactions, catalysis, adsorptions"),
        "dpa3": ("Deep Potential Architecture v3 (alias of deepmd)", "DeepMD legacy alias"),
    }

    lines = [
        "# Supported MLIP Architectures in Janus-Core",
        "",
        "| Architecture (`--arch`) | Description | Typical Use Cases |",
        "| :--- | :--- | :--- |",
    ]
    for a in archs:
        desc, use = arch_details.get(a, ("Machine Learned Interatomic Potential", "General atomistic modeling"))
        lines.append(f"| `{a}` | {desc} | {use} |")

    lines.extend([
        "",
        "## Supported Compute Devices (`--device`)",
        "",
        "- `cpu`: Default fallback; runs on host CPU threads.",
        "- `cuda`: NVIDIA GPU acceleration via CUDA.",
        "- `mps`: Apple Silicon GPU acceleration via Metal Performance Shaders.",
        "- `xpu`: Intel GPU acceleration.",
        "",
        "## Calculator Construction (`--calc-kwargs`)",
        "",
        "Pass calculator-specific keyword arguments as a JSON/Python dictionary string or YAML mapping:",
        "```bash",
        "janus singlepoint --arch mace_mp --calc-kwargs \"{'dispersion': True}\" --struct NaCl.cif",
        "```",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def generate_cli_overview_ref(commands: dict, out_path: Path) -> None:
    """Generate references/cli_overview.md."""
    lines = [
        "# Janus-Core CLI Overview",
        "",
        "The STFC `janus-core` CLI entrypoint is `janus`.",
        "",
        "```text",
        "Usage: janus [OPTIONS] COMMAND [ARGS]...",
        "",
        "Calculations:",
        "  singlepoint   Perform single point calculations and save to file.",
        "  geomopt       Perform geometry optimization and save optimized structure.",
        "  md            Run molecular dynamics simulation (NVE, NVT, NPT).",
        "  phonons       Calculate phonons and save band structures / DOS.",
        "  eos           Calculate equation of state and bulk modulus.",
        "  elasticity    Calculate elasticity tensors.",
        "  neb           Run Nudged Elastic Band method.",
        "  descriptors   Calculate MLIP descriptors.",
        "",
        "Training:",
        "  train         Train or fine-tune an MLIP.",
        "  preprocess    Preprocess data before training.",
        "```",
        "",
        "## Universal Flags Across All Commands",
        "",
        "| Flag | Type | Default | Description |",
        "| :--- | :--- | :--- | :--- |",
        "| `--config` | PATH | None | Path to YAML configuration file supplying CLI options. |",
        "| `--arch` | STRING | mace_mp | Machine Learned Interatomic Potential architecture. |",
        "| `--device` | STRING | cpu | Compute device (`cpu`, `cuda`, `mps`, `xpu`). |",
        "| `--model` | PATH / STR | None | Path to custom checkpoint file or model identifier. |",
        "| `--file-prefix`| PATH | ./janus_results | Directory and filename prefix for outputs. |",
        "| `--no-tracker` | FLAG | (tracker=True) | Disable CodeCarbon emissions tracking. |",
        "| `--log` | PATH | auto | Path to write execution log file. |",
        "| `--summary` | PATH | auto | Path to write execution summary YAML. |",
        "| `--help` | FLAG | None | Display subcommand options and help. |",
        "",
        "## Subcommands Registered in CLI",
        "",
    ]
    for cmd, info in sorted(commands.items()):
        params_str = ", ".join(f"`--{p.replace('_', '-')}`" for p in info.get("params", [])[:6])
        if len(info.get("params", [])) > 6:
            params_str += ", ..."
        lines.append(f"- **`janus {cmd}`**: Key options: {params_str}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def sync_examples(repo_path: Path, output_dir: Path, verbose: bool = False) -> None:
    """Synchronize YAML configuration examples from janus-core docs into skills."""
    docs_examples = repo_path / "docs" / "source" / "examples"
    if not docs_examples.exists():
        log("No docs/source/examples found in repository.", verbose)
        return

    full_dir = docs_examples / "full"
    skills_root = output_dir / "skills"

    mapping = {
        "singlepoint.yml": skills_root / "janus-singlepoint" / "examples" / "singlepoint_config.yml",
        "geomopt.yml": skills_root / "janus-geomopt" / "examples" / "geomopt_config.yml",
        "nvt.yml": skills_root / "janus-md" / "examples" / "nvt_config.yml",
        "npt.yml": skills_root / "janus-md" / "examples" / "npt_config.yml",
        "eos.yml": skills_root / "janus-eos-elasticity" / "examples" / "eos_config.yml",
        "elasticity.yml": skills_root / "janus-eos-elasticity" / "examples" / "elasticity_config.yml",
        "phonons.yml": skills_root / "janus-phonons" / "examples" / "phonons_config.yml",
        "neb.yml": skills_root / "janus-neb" / "examples" / "neb_config.yml",
        "descriptors.yml": skills_root / "janus-descriptors" / "examples" / "descriptors_config.yml",
    }

    copied = 0
    for src_name, dst_path in mapping.items():
        src_path = full_dir / src_name
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied += 1
            log(f"Copied example: {src_name} -> {dst_path.relative_to(output_dir)}", verbose, is_verbose_only=True)

    log(f"Synchronized {copied} example configuration files from docs/source/examples.", verbose)


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
    data["name"] = "janus-core"
    data["version"] = version
    data["description"] = (
        "Comprehensive AI skills, execution rules, and workflow tools for STFC janus-core "
        "(atomistic simulations with MLIPs)."
    )
    data["homepage"] = DEFAULT_DOCS_URL
    data["repository"] = {"type": "git", "url": DEFAULT_REPO_URL}
    data["author"] = {"name": "STFC / Janus-Core Contributors", "url": "https://github.com/stfc/janus-core"}
    plugin_json.write_text(json.dumps(data, indent=2) + "\n")
    log(f"Updated {version_file.name} and {plugin_json.name} with version {version}.", verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Update Janus-Core AI skills and references using upstream GitHub repo and documentation."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_URL,
        help=f"Path to local janus-core clone or Git repository URL (default: {DEFAULT_REPO_URL})",
    )
    parser.add_argument(
        "--docs-url",
        default=DEFAULT_DOCS_URL,
        help=f"Published documentation website URL (default: {DEFAULT_DOCS_URL})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root directory of the janus-core skill group (default: auto-detected)",
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

    # Determine output directory (janus-core group root)
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        # Script is located in <group_root>/scripts/update_skills.py
        output_dir = Path(__file__).resolve().parent.parent

    log(f"Target janus-core skill group directory: {output_dir}")

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

    # 4. Parse upstream CLI and Types
    types_info = parse_types(repo_path)
    commands = parse_cli_commands(repo_path)
    log(f"Parsed {len(types_info.get('architectures', []))} architectures and {len(commands)} CLI commands.")

    if args.check:
        log("=== Check Mode Summary ===")
        log(f"Local Version:    {local_version}")
        log(f"Upstream Version: {upstream_version}")
        log(f"Docs Live:        {docs_info['accessible']} (version: {docs_info['version']})")
        log(f"CLI Commands:     {', '.join(sorted(commands.keys()))}")
        log(f"Architectures:    {', '.join(types_info.get('architectures', []))}")
        if local_version != upstream_version:
            log(f"Update available: {local_version} -> {upstream_version}. Run without --check to apply.")
        else:
            log("Skills are currently up-to-date with upstream.")
        return

    # 5. Apply updates
    log(f"Updating reference documentation and examples for v{upstream_version}...")

    # Reference files
    skills_root = output_dir / "skills"
    core_ref_dir = skills_root / "janus-core" / "references"
    generate_architectures_ref(types_info, core_ref_dir / "architectures.md")
    generate_cli_overview_ref(commands, core_ref_dir / "cli_overview.md")

    # Sync example configurations
    sync_examples(repo_path, output_dir, verbose=args.verbose)

    # Update VERSION and plugin.json
    update_version_and_plugin(upstream_version, output_dir, verbose=args.verbose)

    log("✓ Successfully updated Janus-Core skills, reference files, and examples.")


if __name__ == "__main__":
    main()

