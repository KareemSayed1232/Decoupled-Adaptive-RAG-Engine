import os

# Folders/files to ignore
IGNORE_DIRS = {
    "__pycache__", ".git", ".idea", ".vscode", "node_modules", ".mypy_cache", ".pytest_cache"
}
IGNORE_FILES = {
    ".DS_Store", "Thumbs.db"
}

def print_tree(start_path=".", prefix=""):
    # Sort to get consistent order
    entries = sorted(os.listdir(start_path))
    
    # Filter out ignored dirs and files
    entries = [
        e for e in entries
        if e not in IGNORE_FILES and e not in IGNORE_DIRS
    ]

    for index, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = "└── " if index == len(entries) - 1 else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "    " if index == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print_tree(".")