"""
Generate the expert hierarchy as markdown files.
Each file = one expert node with tags defining its domain/path.
"""
import os

NODES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes")

EXPERTS = {
    "core": {
        "tags": ["general", "reasoning", "planning", "language"],
        "desc": "Core brain — general reasoning and coordination",
    },
    "coding": {
        "tags": ["coding", "programming", "software"],
        "desc": "General coding knowledge",
        "children": {
            "python": {
                "tags": ["coding", "python"],
                "desc": "Python language fundamentals",
                "children": {
                    "python_api":    {"tags": ["coding", "python", "api"],    "desc": "Python API development"},
                    "python_ai_ml":  {"tags": ["coding", "python", "ai"],     "desc": "Python AI/ML"},
                    "python_async":  {"tags": ["coding", "python", "async"],  "desc": "Python async/concurrency"},
                    "python_test":   {"tags": ["coding", "python", "test"],   "desc": "Python testing"},
                },
            },
            "rust": {
                "tags": ["coding", "rust"],
                "desc": "Rust language",
                "children": {
                    "rust_systems":  {"tags": ["coding", "rust", "systems"], "desc": "Rust systems programming"},
                    "rust_web":      {"tags": ["coding", "rust", "web"],     "desc": "Rust web development"},
                },
            },
            "javascript": {
                "tags": ["coding", "javascript"],
                "desc": "JavaScript/TypeScript",
                "children": {
                    "js_react":      {"tags": ["coding", "javascript", "react"],    "desc": "React frontend"},
                    "js_node":       {"tags": ["coding", "javascript", "node"],      "desc": "Node.js backend"},
                    "js_typescript": {"tags": ["coding", "javascript", "typescript"], "desc": "TypeScript"},
                },
            },
            "go": {
                "tags": ["coding", "go"],
                "desc": "Go language",
            },
        },
    },
    "math": {
        "tags": ["math", "mathematics"],
        "desc": "Mathematics",
        "children": {
            "math_algebra":    {"tags": ["math", "algebra"],    "desc": "Algebra"},
            "math_calculus":   {"tags": ["math", "calculus"],   "desc": "Calculus"},
            "math_statistics": {"tags": ["math", "statistics"], "desc": "Statistics & probability"},
            "math_geometry":   {"tags": ["math", "geometry"],   "desc": "Geometry"},
        },
    },
    "science": {
        "tags": ["science"],
        "desc": "General science",
        "children": {
            "science_physics":   {"tags": ["science", "physics"],   "desc": "Physics"},
            "science_chemistry": {"tags": ["science", "chemistry"], "desc": "Chemistry"},
            "science_biology":   {"tags": ["science", "biology"],   "desc": "Biology"},
        },
    },
    "language": {
        "tags": ["language", "writing"],
        "desc": "Language & writing",
        "children": {
            "lang_grammar":    {"tags": ["language", "grammar"],    "desc": "Grammar & style"},
            "lang_writing":    {"tags": ["language", "writing"],    "desc": "Creative & technical writing"},
            "lang_translate":  {"tags": ["language", "translation"], "desc": "Translation"},
        },
    },
}


def write_node(path: str, node_id: str, data: dict):
    tags = " ".join(f"#{t}" for t in data["tags"])
    content = f"# {node_id}\n# {data['desc']}\n{tags}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def build_tree(base_dir: str, tree: dict, parent_tags: list | None = None):
    parent_tags = parent_tags or []
    for node_id, data in tree.items():
        tags = parent_tags + data["tags"]
        file_path = os.path.join(base_dir, f"{node_id}.md")
        write_node(file_path, node_id, data)
        print(f"  {file_path}  tags={tags}")
        children = data.get("children", {})
        if children:
            child_dir = os.path.join(base_dir, node_id)
            os.makedirs(child_dir, exist_ok=True)
            build_tree(child_dir, children, tags)


if __name__ == "__main__":
    os.makedirs(NODES_DIR, exist_ok=True)
    print(f"Generating experts in {NODES_DIR}...")
    build_tree(NODES_DIR, EXPERTS)
    print(f"\nDone. {sum(1 for _, _, files in os.walk(NODES_DIR) for f in files if f.endswith('.md'))} expert files created.")
