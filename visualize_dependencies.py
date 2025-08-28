# In visualize_dependencies.py
import os
import ast
from graphviz import Digraph
from collections import defaultdict
import sys

print("--- Decoupled Project Dependency Visualizer ---")

# --- CONFIGURATION ---
PROJECT_ROOT = '.'
OUTPUT_FILENAME = 'decoupled_dependency_graph'
PYTHON_EXTENSIONS = {'.py'}

# Define our services and packages to create clusters
# The key is the directory path, the value is the display name and color
SERVICE_CLUSTERS = {
    'services/rag_api': ('RAG API (The Brains)', 'lightblue'),
    'services/inference_api': ('Inference API (The Brawn)', 'lightgreen'),
    'packages/shared-models': ('Shared Models', 'gold'),
    'clients/gradio-demo': ('Gradio Client', 'lightpink'),
}

# --- HELPER FUNCTIONS ---

def find_py_files(root):
    """Finds all Python files in the project."""
    for dirpath, _, files in os.walk(root):
        if 'site-packages' in dirpath or '.venv' in dirpath:
            continue
        for f in files:
            if os.path.splitext(f)[1] in PYTHON_EXTENSIONS:
                yield os.path.join(dirpath, f)

def module_name_from_path(path):
    """Converts a file path to a Python module path (e.g., services.rag_api.src.main)."""
    rel_path = os.path.relpath(path, PROJECT_ROOT)
    no_ext = os.path.splitext(rel_path)[0]
    # Replace OS-specific separators with dots
    parts = no_ext.replace(os.sep, '.').split('.')
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return ".".join(parts)

def get_cluster_key(module_path):
    """Determines which service/package a module belongs to."""
    for service_path in SERVICE_CLUSTERS:
        service_module_path = service_path.replace(os.sep, '.')
        if module_path.startswith(service_module_path):
            return service_path
    return None

def extract_imports(filepath):
    """Parses a Python file and extracts all its import statements."""
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content, filename=filepath)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:  # Absolute imports only
                    imports.add(node.module)
    except Exception as e:
        print(f"  - WARNING: Could not parse '{filepath}': {e}")
    return imports

# --- MAIN SCRIPT LOGIC ---

def build_dependency_graph():
    """Builds and renders the dependency graph."""
    # Add project root to path to allow parsing of absolute imports
    sys.path.insert(0, os.path.abspath(PROJECT_ROOT))
    
    # 1. Find all Python files and map them to module names
    files = list(find_py_files(PROJECT_ROOT))
    module_map = {module_name_from_path(f): f for f in files}

    # 2. Group modules by their service/package
    grouped_modules = defaultdict(list)
    for mod in module_map.keys():
        cluster_key = get_cluster_key(mod)
        if cluster_key:
            grouped_modules[cluster_key].append(mod)

    # 3. Initialize the graph
    graph = Digraph(comment='Decoupled Project Dependency Graph')
    graph.attr(rankdir='LR', splines='ortho', nodesep='0.5', ranksep='1.2', compound='true')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    graph.attr('edge', fontname='Helvetica', fontsize='10')

    # 4. Create a subgraph (cluster) for each service/package
    print("\n[1/3] Creating service and package clusters...")
    for cluster_key, (label, color) in SERVICE_CLUSTERS.items():
        print(f"  - Creating cluster for '{label}'")
        with graph.subgraph(name=f'cluster_{label.replace(" ", "_")}') as sub:
            sub.attr(label=label, style='filled', color=color)
            for mod in grouped_modules[cluster_key]:
                # Shorten the label for display (e.g., 'src.main' instead of full path)
                short_label = mod.replace(cluster_key.replace(os.sep, '.'), '').lstrip('.')
                sub.node(mod, short_label)

    # 5. Add edges based on imports
    print("\n[2/3] Analyzing imports and adding dependencies...")
    external_deps = set()
    for mod, filepath in module_map.items():
        imports = extract_imports(filepath)
        for imp in imports:
            # Check for dependencies on our own modules
            found_internal_dep = False
            for internal_mod in module_map:
                if internal_mod == imp or imp.startswith(internal_mod + "."):
                    # Don't draw an edge from a module to itself
                    if mod != internal_mod:
                        graph.edge(mod, internal_mod)
                        print(f"  - Found dependency: {mod} -> {internal_mod}")
                    found_internal_dep = True
                    break
            
            # Track external dependencies (like fastapi, gradio, etc.)
            if not found_internal_dep:
                external_deps.add(imp.split('.')[0])
    
    # 6. (Optional) Create a node for external dependencies
    with graph.subgraph(name='cluster_external') as sub:
        sub.attr(label='External Libraries', style='dotted')
        for dep in sorted(list(external_deps)):
             if dep not in ('services', 'packages', 'clients'): # Exclude our own top-level packages
                sub.node(dep, dep, shape='ellipse', color='lightgrey')
                
    return graph

if __name__ == '__main__':
    try:
        dependency_graph = build_dependency_graph()
        
        print(f"\n[3/3] Rendering graph to '{OUTPUT_FILENAME}.png'...")
        # The 'view=True' argument will try to automatically open the generated image.
        dependency_graph.render(OUTPUT_FILENAME, view=True, format='png', cleanup=True)
        
        print("\n" + "="*50)
        print("✅ Visualization complete!")
        print(f"File saved as '{OUTPUT_FILENAME}.png'")
        print("="*50)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("\n--- Troubleshooting ---")
        print("1. Make sure you have run 'pip install graphviz'.")
        print("2. Make sure you have installed the Graphviz system software and that it's in your system's PATH.")
        print("   - For Windows, download from the official site.")
        print("   - For Linux, use 'sudo apt-get install graphviz'.")
        print("   - For macOS, use 'brew install graphviz'.")
        import traceback
        traceback.print_exc()