import os
import json
def generate_yaml_files(pairs, output_dir="."):
    """
    Generate YAML files for given (lang, domain) pairs.

    Args:
        pairs (list of tuple): List of (lang, domain) pairs.
        output_dir (str): Directory where YAML files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    template = (
        "include: _rubric_rating.yaml\n\n"
        "task: rubric_rating_{lang}_{domain}\n\n"
        "task_alias: {lang}_{domain}\n\n"
        "process_docs: !function utils.process_{lang}_{domain}\n"
    )

    for lang, domain in pairs:
        content = template.format(lang=lang, domain=domain)
        filename = f"rubric_rating_{lang}_{domain}.yaml"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Generated: {filepath}")

if __name__ == "__main__":
    import argparse
    arser = argparse.ArgumentParser(description="Generate YAML files for (lang, domain) pairs.")
    arser.add_argument("--output_dir", type=str, default=".", help="Directory to save generated YAML files.")
    arser.add_argument("--dataset_file", type=str, default="./dev.rubric.json", help="Path to the dataset file.")
    args = arser.parse_args()
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = sorted(set((x["lang"], x["domain"]) for x in data))
    print(pairs)
    generate_yaml_files(pairs)
