import os

def pack_folder(folder_name, output_file):
    print(f"Packing {folder_name}...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(folder_name):
            # Ignore hidden directories, virtual environments, and caches
            if any(part.startswith('.') for part in root.split(os.sep)) or 'venv' in root or '__pycache__' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    outfile.write(f"\n--- START FILE: {file_path} ---\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"# Error reading file: {e}\n")
                    outfile.write(f"\n--- END FILE: {file_path} ---\n")

if __name__ == "__main__":
    pack_folder('BeatSync-Engine', 'beatsync_source.txt')
    pack_folder('auto-gaming-montage-maker', 'montage_fx_source.txt')
    print("Success! Generated beatsync_source.txt and montage_fx_source.txt")
