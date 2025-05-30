#!/usr/bin/env python3
import subprocess
import sys

def run(cmd: list[str]) -> None:
    """Execute a command and raise if it fails."""
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # Lancer phase4.py
    run([
        sys.executable,
        "phase4.py",
        "--config", "config.yaml",
        "--datasets", "raw", "cleaned_1", "cleaned_3_multi", "cleaned_3_univ"
    ])

    # Puis générer le rapport PDF
    run([
        sys.executable,
        "generate_phase4_report.py"
    ])

    print("\n✅ Tout s’est bien déroulé !")

if __name__ == "__main__":
    main()
