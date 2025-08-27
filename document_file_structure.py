import os
import pathlib

def parse_gitignore(git_ignore_path):
    """
    Parses a .gitignore file and returns a list of patterns to ignore.
    """
    if not os.path.exists(git_ignore_path):
        return []

    with open(git_ignore_path, 'r') as f:
        # Filter out comments and empty lines
        patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return patterns

def is_ignored(path, gitignore_patterns, root_path):
    """
    Checks if a given path should be ignored based on .gitignore patterns.
    """
    # Normalize path to be relative to the root
    relative_path = os.path.relpath(path, root_path)

    for pattern in gitignore_patterns:
        # Handle patterns that match entire directories
        if pattern.endswith('/'):
            if relative_path.startswith(pattern.rstrip('/')):
                return True
        # Handle patterns for specific files
        elif pathlib.PurePath(relative_path).match(pattern) or pathlib.PurePath(os.path.basename(relative_path)).match(pattern):
            return True
            
    return False

def document_file_structure(start_path, output_file='project_structure.txt'):
    """
    Documents the file structure of a project, respecting .gitignore rules.
    """
    gitignore_path = os.path.join(start_path, '.gitignore')
    gitignore_patterns = parse_gitignore(gitignore_path)

    output_filename = os.path.basename(output_file)
    if output_filename not in gitignore_patterns:
        gitignore_patterns.append(output_filename)

    with open(output_file, 'w') as f:
        f.write(f"{os.path.basename(start_path)}/\n")

        for root, dirs, files in os.walk(start_path):
            dirs.sort()
            files.sort()

            level = root.replace(start_path, '').count(os.sep)
            indent = '    ' * (level)

            prune_dirs = []
            for d in dirs:
                full_path = os.path.join(root, d)
                if is_ignored(full_path, gitignore_patterns, start_path):
                    prune_dirs.append(d)
            
            for d in prune_dirs:
                dirs.remove(d)

            for d in dirs:
                f.write(f"{indent}|-- {d}/\n")

            prune_files = []
            for file in files:
                full_path = os.path.join(root, file)
                if is_ignored(full_path, gitignore_patterns, start_path):
                    prune_files.append(file)
            
            for file in prune_files:
                files.remove(file)
            
            for file in files:
                f.write(f"{indent}|-- {file}\n")
    
    print(f"File structure documented in {output_file}")


if __name__ == '__main__':
    project_root = os.getcwd()
    document_file_structure(project_root)