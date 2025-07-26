import os
import json
import re
from glob import glob

def _read_txt_content(txt_path: str) -> str:

    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1].strip()

    content = content.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').strip()
    content = re.sub(r'([.!?])\s*', r'\1\n', content)
    
    content = re.sub(r'\n+', r'\n', content)
    
    content = content.strip()
    
    return content

def create_character_json(txt_path: str, anime: str) -> dict:
    """
    Converts a character TXT file to JSON format
    """
    content = _read_txt_content(txt_path)
    name = os.path.basename(txt_path).replace('.txt', '')

    # for more advanced metadata extraction or manual input
    metadata = {
        "source_file": os.path.basename(txt_path),
        "traits": [], 
        "archetype": "", 
        "narrative_function": "", 
        "significant_locations": [] 
    }

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
        "metadata": metadata
    }

def create_episode_json(txt_path: str, anime: str) -> dict:
    """
    Converts an episode TXT file to JSON format
    """
    content = _read_txt_content(txt_path)
    filename = os.path.basename(txt_path).replace('.txt', '')

    title = ""
    season = ""
    episode_num = ""

    if anime == "aot":
        #  AOT format: Season_3_episode_2.txt
        parts = filename.split('_')
        # ['Season', 'SEASON_NUMBER', 'episode', 'EPISODE_NUMBER']
        if len(parts) == 4 and parts[0] == 'Season' and parts[2] == 'episode':
            season = parts[1]
            episode_num = parts[3]
            title = f"Attack on Titan Season {season} Episode {episode_num}"
        else:
            print(f"Warning: wrong AOT episode filename format: {filename}")
            season = "Unknown"
            episode_num = filename 
            title = f"Attack on Titan Episode {filename}"
    elif anime == "fmab":
        #  FMAB format: episode_39.txt
        parts = filename.split('_')
        if len(parts) == 2 and parts[0] == 'episode':
            episode_num = parts[1]
            title = f"Fullmetal Alchemist Brotherhood Episode {episode_num}"
            season = "1"
        else:
            print(f"Warning: wrong FMAB episode filename format: {filename}.")
            season = "1"
            episode_num = filename
            title = f"Fullmetal Alchemist Brotherhood Episode {filename}"
    else:
        print(f"Warning: wrong anime '{anime}' for episode processing. Using normal naming.")
        title = filename
        season = "Unknown"
        episode_num = filename

    # for more advanced metadata extraction
    metadata = {
        "source_file": os.path.basename(txt_path),
        "themes": [] 
    }

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
        "metadata": metadata
    }

def process_folder(root_dir: str, output_dir: str):
    """
    Processes all TXT files in the specified directory structure, converting them to JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    anime_types = ['aot', 'fmab']

    for anime in anime_types:
        char_dir = os.path.join(root_dir, 'characters', anime)
        if not os.path.exists(char_dir):
            continue

        for txt_file in glob(os.path.join(char_dir, '*.txt')):
            try:
                json_data = create_character_json(txt_file, anime)
                output_file = os.path.join(output_dir, f"char_{anime}_{os.path.basename(txt_file).replace('.txt', '.json')}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error processing character file {txt_file}: {e}")

    for anime in anime_types:
        ep_dir = os.path.join(root_dir, 'episodes', anime)
        if not os.path.exists(ep_dir):
            continue

        for txt_file in glob(os.path.join(ep_dir, '*.txt')):
            try:
                json_data = create_episode_json(txt_file, anime)
                output_file = os.path.join(output_dir, f"ep_{anime}_{os.path.basename(txt_file).replace('.txt', '.json')}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"Error processing episode file {txt_file}: {e}")

if __name__ == "__main__":
    RAW_DATA_DIR = "./raw_data"
    OUTPUT_DIR = "./json_output"
    
    process_folder(RAW_DATA_DIR, OUTPUT_DIR)
    print(f"\n DONE!!. JSON files saved to {OUTPUT_DIR}")