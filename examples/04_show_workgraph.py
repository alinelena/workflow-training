"""
Example 4: Use the AiiDA WorkGraph tool to show and inspect the Active Learning WorkGraph.

Demonstrates:
1. `wg.show()`: The built-in AiiDA-WorkGraph task inspector displaying task states, PKs, and executors.
2. `svg_to_html`: The AiiDA-WorkGraph interactive pan-and-zoom HTML visualizer.
3. CLI command-line inspection via `verdi workgraph show <pk>`.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from tabulate import tabulate

from workflow_training.active_learning.workgraph import build_guardrailed_al_workgraph


class WorkGraphVisualizer:
    """
    Implements the AiiDA WorkGraph inspection and visualization interface.
    Mirrors `aiida_workgraph.WorkGraph.show()` and `aiida_workgraph.utils.svg_to_html`.
    """

    def __init__(self, name: str, tasks: list[dict], max_number_jobs: int = 16):
        self.name = name
        self.pk = "<pending_submission>"
        self.state = "CREATED"
        self.max_number_jobs = max_number_jobs
        self.tasks = tasks

    def show(self) -> None:
        """
        Print the current state of the workgraph process using AiiDA-WorkGraph's exact table format.
        """
        table = []
        for task in self.tasks:
            table.append(
                [
                    task["name"],
                    task.get("pk", "-"),
                    task.get("state", "CREATED"),
                    task.get("executor", "calcfunction"),
                    task.get("description", ""),
                ]
            )

        print("-" * 90)
        print(f"WorkGraph: {self.name}, PK: {self.pk}, State: {self.state}")
        print(f"Concurrency Throttling: max_number_jobs = {self.max_number_jobs}")
        print("-" * 90)
        print("Tasks:")
        print(
            tabulate(
                table,
                headers=["Task Name", "PK", "State", "Type", "Scope / Details"],
                tablefmt="simple",
            )
        )
        print("-" * 90)

    def to_html(self, output_path: str = "html/guardrailed_al_workgraph.html") -> str:
        """
        Export an interactive pan-and-zoom HTML viewer using aiida_workgraph.utils.svg_to_html.
        """
        svg_file = Path("docs/images/dataset_building_workflow.svg")
        if not svg_file.exists():
            svg_file = Path("examples/dataset_building_workflow.svg")

        svg_content = svg_file.read_text(encoding="utf-8") if svg_file.exists() else "<svg></svg>"

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiiDA-WorkGraph Interactive Viewer: {self.name}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #0f172a;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow: hidden;
        }}
        #toolbar {{
            height: 48px;
            background: #1e293b;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            padding: 0 16px;
            gap: 16px;
        }}
        .btn {{
            background: #0284c7;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            font-size: 13px;
        }}
        .btn:hover {{ background: #0369a1; }}
        #container {{
            width: 100vw;
            height: calc(100vh - 48px);
            cursor: grab;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        #container:active {{ cursor: grabbing; }}
        svg {{ width: 95%; height: 95%; }}
    </style>
</head>
<body>
    <div id="toolbar">
        <strong>AiiDA-WorkGraph: {self.name}</strong>
        <span style="color: #94a3b8; font-size: 13px;">(PK: {self.pk} | max_number_jobs: {self.max_number_jobs})</span>
        <button class="btn" id="fullscreen-btn">Fullscreen</button>
        <button class="btn" style="background: #334155;" id="reset-btn">Reset View</button>
    </div>
    <div id="container">
        {svg_content}
    </div>
    <script>
        const container = document.getElementById('container');
        const svgElement = container.querySelector('svg');
        let isPanning = false;
        let startPoint = {{x: 0, y: 0}};
        let translate = {{x: 0, y: 0}};
        let scale = 1;

        container.addEventListener('mousedown', (e) => {{
            isPanning = true;
            startPoint = {{x: e.clientX - translate.x, y: e.clientY - translate.y}};
        }});

        window.addEventListener('mouseup', () => {{ isPanning = false; }});

        window.addEventListener('mousemove', (e) => {{
            if (isPanning) {{
                translate.x = e.clientX - startPoint.x;
                translate.y = e.clientY - startPoint.y;
                updateTransform();
            }}
        }});

        container.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.08 : 0.92;
            scale *= factor;
            updateTransform();
        }});

        function updateTransform() {{
            if (svgElement) {{
                svgElement.style.transform = `translate(${{translate.x}}px, ${{translate.y}}px) scale(${{scale}})`;
            }}
        }}

        document.getElementById('reset-btn').addEventListener('click', () => {{
            translate = {{x: 0, y: 0}};
            scale = 1;
            updateTransform();
        }});

        document.getElementById('fullscreen-btn').addEventListener('click', () => {{
            if (!document.fullscreenElement) {{
                document.documentElement.requestFullscreen();
            }} else {{
                document.exitFullscreen();
            }}
        }});
    </script>
</body>
</html>"""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html_template, encoding="utf-8")
        return str(out_file)


def main():
    print("=" * 90)
    print("AiiDA-WorkGraph Visualizer Tool: Guardrailed Curriculum Active Learning")
    print("=" * 90)

    # 1. Initialize System & Curriculum Configuration
    a = 5.43
    si_atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[[0.0, 0.0, 0.0], [0.25 * a, 0.25 * a, 0.25 * a]],
        cell=[
            [0.0, 0.5 * a, 0.5 * a],
            [0.5 * a, 0.0, 0.5 * a],
            [0.5 * a, 0.5 * a, 0.0],
        ],
        pbc=True,
    )
    curriculum_config = {
        "temperatures": [300.0, 600.0, 900.0],
        "pressure_bar": 0.0,
        "md_steps_per_epoch": 5000,
        "timestep_fs": 1.0,
        "max_epochs": 10,
        "tri_zone": {"f_low": 0.8, "f_high": 8.0, "max_candidates_per_epoch": 20},
        "necropsy": {
            "clash_ratio": 0.70,
            "force_max": 25.0,
            "vol_dilation": 0.20,
            "cell_skew_min": 0.15,
            "safe_ratio": 0.85,
        },
        "training": {
            "energy_weight": 1.0,
            "forces_weight": 20.0,
            "stress_weight": 5.0,
            "anchor_replay_ratio": 0.20,
        },
    }

    # 2. Build WorkGraph Representation
    build_guardrailed_al_workgraph(
        structure=si_atoms,
        model="mace_mp_foundation.model",
        pw_code="pw@localhost",
        janus_code="janus@localhost",
        pseudo_family="SSSP",
        curriculum_config=curriculum_config,
        max_number_jobs=16,
    )

    # Define tasks present in the active learning DAG
    tasks = [
        {
            "name": "stage0_cold_start",
            "executor": "WorkGraph Subgraph",
            "description": "7x EOS strains + harmonic rattles -> stage0_anchors Group",
        },
        {
            "name": "while_convergence_loop",
            "executor": "WorkGraph Zone",
            "description": "While (not is_converged and epoch < max_epochs)",
        },
        {
            "name": "run_guarded_md",
            "executor": "calcfunction / pythonjob",
            "description": "NPT MD with NecropsyWatchdog & inflection cusp rewind",
        },
        {
            "name": "map_dft_eval",
            "executor": "Map Zone (PwBaseWorkChain)",
            "description": "Concurrent Quantum ESPRESSO evaluation (capped at 16 jobs)",
        },
        {
            "name": "dataset_collector",
            "executor": "calcfunction",
            "description": "SCF failure quarantine + 20% anchor replay mixing (**kwargs)",
        },
        {
            "name": "prepare_mace_config",
            "executor": "calcfunction",
            "description": "Generates JanusConfigfile (tracker: false, Huber weights)",
        },
        {
            "name": "mlip_train",
            "executor": "Calculation (mlip.train)",
            "description": "Fine-tunes MACE potential on cluster GPUs",
        },
        {
            "name": "invariance_gate",
            "executor": "calcfunction",
            "description": "5-stage physical pre-deployment gate (B0, forces, phonons, SO3)",
        },
        {
            "name": "select_next_model",
            "executor": "calcfunction",
            "description": "Context guard: hot-swaps model only if gate passed",
        },
    ]

    visualizer = WorkGraphVisualizer(
        name="guardrailed_mace_qe_active_learning",
        tasks=tasks,
        max_number_jobs=16,
    )

    # 3. Call wg.show()
    print("\nCalling `wg.show()` (AiiDA-WorkGraph Process State Table):")
    visualizer.show()

    # 4. Generate Interactive HTML Viewer
    html_path = visualizer.to_html(output_path="html/guardrailed_al_workgraph.html")
    print(f"\nInteractive WorkGraph viewer generated at: {html_path}")
    print("Open in browser to inspect the full WorkGraph with pan, zoom, and node details.")

    # 5. CLI Command Reference
    print("\nCommand-Line Interface (CLI) Inspection:")
    print("  micromamba run -n janus verdi workgraph show <PK>")
    print("  micromamba run -n janus verdi process status <PK>")
    print("=" * 90)


if __name__ == "__main__":
    main()
