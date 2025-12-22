import os
import json
import re
from glob import glob

def read_txt(txt_path: str) -> str:
    """Read a TXT file and clean up content."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Remove surrounding quotes if present
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1].strip()

    # Normalize newlines and split sentences
    content = content.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').strip()
    content = re.sub(r'([.!?])\s*', r'\1\n', content)
    content = re.sub(r'\n+', r'\n', content)

    return content

def create_character_json(txt_path: str, anime: str) -> dict:
    #Convert a character TXT file to JSON.
    content = read_txt(txt_path)
    name = os.path.basename(txt_path).replace('.txt', '')

    return {
        "name": name,
        "anime": anime,
        "type": "character",
        "aliases": [],
        "gender": "",
        "affiliations": [],
        "family_members": [],
        "key_relationships": [],
        "abilities": [],
        "status": "",
        "content": content,
        "metadata": {
            "source_file": os.path.basename(txt_path),
            "traits": [],
            "archetype": "",
            "narrative_function": "",
            "significant_locations": []
        }
    }

def create_episode_json(txt_path: str, anime: str) -> dict:
    #Converts an episode TXT file to JSON
    content = read_txt(txt_path)
    filename = os.path.basename(txt_path).replace('.txt', '')

    # Default values
    title = filename
    season = "Unknown"
    episode_num = filename

    # Parse filenames based on anime
    if anime == "aot":
        parts = filename.split('_')
        if len(parts) == 4 and parts[0] == 'Season' and parts[2] == 'episode':
            season = parts[1]
            episode_num = parts[3]
            title = f"Attack on Titan Season {season} Episode {episode_num}"
    elif anime == "fmab":
        parts = filename.split('_')
        if len(parts) == 2 and parts[0] == 'episode':
            season = "1"
            episode_num = parts[1]
            title = f"Fullmetal Alchemist Brotherhood Episode {episode_num}"

    return {
        "title": title,
        "anime": anime,
        "type": "episode",
        "season": season,
        "episode_number": episode_num,
        "arc": "",
        "key_events": [],
        "featured_characters": [],
        "content": content,
        "metadata": {
            "source_file": os.path.basename(txt_path),
            "themes": []
        }
    }

def process_folder(root_dir: str, output_dir: str):
    #Process all TXT files into JSON
    os.makedirs(output_dir, exist_ok=True)
    anime_types = ['aot', 'fmab']

    for anime in anime_types:
        # Characters
        char_dir = os.path.join(root_dir, 'characters', anime)
        for txt_file in glob(os.path.join(char_dir, '*.txt')):
            json_data = create_character_json(txt_file, anime)
            out_file = os.path.join(output_dir, f"char_{anime}_{os.path.basename(txt_file).replace('.txt', '.json')}")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Episodes
        ep_dir = os.path.join(root_dir, 'episodes', anime)
        for txt_file in glob(os.path.join(ep_dir, '*.txt')):
            json_data = create_episode_json(txt_file, anime)
            out_file = os.path.join(output_dir, f"ep_{anime}_{os.path.basename(txt_file).replace('.txt', '.json')}")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    RAW_DATA_DIR = "./raw_data"
    OUTPUT_DIR = "./json_output"
    process_folder(RAW_DATA_DIR, OUTPUT_DIR)
    print(f"\nDONE! JSON files saved to {OUTPUT_DIR}")
