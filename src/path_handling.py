from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_path(path_str):
    p = Path(path_str)
    
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    return p.resolve()

print(f"All relative paths in resolve_path are considered as relative to PROJECT_ROOT directory")