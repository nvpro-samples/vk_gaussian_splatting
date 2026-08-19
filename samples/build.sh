#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── File downloads ───────────────────────────────────────────────────────────
# Entries: "<URL> <local_filename> [subfolder]"
#   subfolder : optional — download into data/<subfolder>/ instead of data/
ASSETS=(
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/horse.obj horse.obj"
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/teapot.obj teapot.obj"
)

# ── Google Drive downloads ────────────────────────────────────────────────────
# Entries: "<file_id> <local_filename> [subfolder]"
#   subfolder : optional — download into data/<subfolder>/ instead of data/
GDRIVE=(
    "1_Gvh6eRjUY1RU4-GCzROFHUeoAj7G-xT 20K-Photo-103Mspats-4x2KM-Andrii_Shramko_Poland-JG.ply teleportour"
    "1UN79ygHHIsM8AHG1FkzI88gnkAG78tNt Winter-Garden-view2.ply teleportour"
    "12TLqEsp-LZBkg8YwddpwpC3i5e5mFChB License_Agreement.md teleportour"
)

# ── Zip downloads (download + extract) ───────────────────────────────────────
# Entries: "<URL> <dest_folder> [mkdir] [subfolder]"
#   dest_folder : name used to detect "already extracted" and as unzip target
#   mkdir       : optional flag — if present, extract into a created dest_folder
#                 (use when the zip has no top-level folder)
#   subfolder   : optional — extract under data/<subfolder>/ instead of data/
ZIPS=(
    "https://developer.download.nvidia.com/ProGraphics/nvpro-samples/flowers_1.zip flowers_1"
    "https://developer.download.nvidia.com/ProGraphics/nvpro-samples/vkgs/astronomical_quintant.zip astronomical_quintant"
)

# ── Google Drive zip downloads (download + extract) ──────────────────────────
# Entries: "<file_id> <dest_folder> [mkdir] [subfolder]"
#   dest_folder : name used to detect "already extracted" and as unzip target
#   mkdir       : optional flag — if present, extract into a created dest_folder
#                 (use when the zip has no top-level folder)
#   subfolder   : optional — extract under data/<subfolder>/ instead of data/
GDRIVE_ZIPS=(
    # "FILE_ID folder_name"
    # "FILE_ID folder_name mkdir"
)

# ── Git sparse-checkout repos ────────────────────────────────────────────────
# Entries: "<repo_url> <commit> <dest_folder> <path1,path2,...>"
GIT_REPOS=(
    "https://github.com/KhronosGroup/glTF-Sample-Assets.git 5109ab2a499c5a2c784b86e460fa491d52256e25 glTF-Sample-Assets Models/ABeautifulGame,Models/FlightHelmet,Models/DiffuseTransmissionTeacup"
)

# ── Resolve destination ──────────────────────────────────────────────────────
DEST_DIR="${1:-$SCRIPT_DIR}"
DEST_DIR="$(cd "$DEST_DIR" 2>/dev/null && pwd || echo "$DEST_DIR")"

if [ "$DEST_DIR" != "$SCRIPT_DIR" ]; then
    mkdir -p "$DEST_DIR"

    echo "Copying .vkgs projects to $DEST_DIR ..."
    for f in "$SCRIPT_DIR"/*.vkgs; do
        [ -e "$f" ] || continue
        cp -v "$f" "$DEST_DIR/"
    done
fi

DATA_DIR="$DEST_DIR/data"
mkdir -p "$DATA_DIR"

# ── Download file assets ─────────────────────────────────────────────────────
for entry in "${ASSETS[@]}"; do
    read -r url filename subfolder <<< "$entry"
    target_dir="$DATA_DIR${subfolder:+/$subfolder}"
    mkdir -p "$target_dir"
    dest_file="$target_dir/$filename"

    if [ -f "$dest_file" ]; then
        echo "Already downloaded: ${subfolder:+$subfolder/}$filename"
        continue
    fi

    echo "Downloading ${subfolder:+$subfolder/}$filename ..."
    curl -fSL --progress-bar -o "$dest_file" "$url"
done

# ── Download Google Drive assets ──────────────────────────────────────────────
for entry in "${GDRIVE[@]}"; do
    read -r file_id filename subfolder <<< "$entry"
    target_dir="$DATA_DIR${subfolder:+/$subfolder}"
    mkdir -p "$target_dir"
    dest_file="$target_dir/$filename"

    if [ -f "$dest_file" ]; then
        echo "Already downloaded: ${subfolder:+$subfolder/}$filename"
        continue
    fi

    echo "Downloading ${subfolder:+$subfolder/}$filename from Google Drive ..."
    curl -fSL --progress-bar -o "$dest_file" \
        "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"
done

# ── Download and extract Google Drive zips ────────────────────────────────────
for entry in "${GDRIVE_ZIPS[@]}"; do
    read -r file_id dest_folder opt1 opt2 <<< "$entry"
    force_mkdir=""; subfolder=""
    for opt in "$opt1" "$opt2"; do
        [ "$opt" = "mkdir" ] && force_mkdir="mkdir" || [ -n "$opt" ] && subfolder="$opt"
    done
    extract_dir="$DATA_DIR${subfolder:+/$subfolder}"
    dest="$extract_dir/$dest_folder"

    if [ -d "$dest" ]; then
        echo "Already extracted: ${subfolder:+$subfolder/}$dest_folder"
        continue
    fi

    mkdir -p "$extract_dir"
    zipfile="$extract_dir/${dest_folder}.zip"
    if [ ! -f "$zipfile" ]; then
        echo "Downloading ${dest_folder}.zip from Google Drive ..."
        curl -fSL --progress-bar -o "$zipfile" \
            "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"
    fi

    echo "Extracting ${subfolder:+$subfolder/}$dest_folder ..."
    if [ "$force_mkdir" = "mkdir" ]; then
        mkdir -p "$dest"
        unzip -qo "$zipfile" -d "$dest"
    else
        unzip -qo "$zipfile" -d "$extract_dir"
    fi

    rm -f "$zipfile"
done

# ── Download and extract zips ─────────────────────────────────────────────────
for entry in "${ZIPS[@]}"; do
    read -r url dest_folder opt1 opt2 <<< "$entry"
    force_mkdir=""; subfolder=""
    for opt in "$opt1" "$opt2"; do
        [ "$opt" = "mkdir" ] && force_mkdir="mkdir" || [ -n "$opt" ] && subfolder="$opt"
    done
    extract_dir="$DATA_DIR${subfolder:+/$subfolder}"
    dest="$extract_dir/$dest_folder"

    if [ -d "$dest" ]; then
        echo "Already extracted: ${subfolder:+$subfolder/}$dest_folder"
        continue
    fi

    mkdir -p "$extract_dir"
    zipfile="$extract_dir/${url##*/}"
    if [ ! -f "$zipfile" ]; then
        echo "Downloading ${url##*/} ..."
        curl -fSL --progress-bar -o "$zipfile" "$url"
    fi

    echo "Extracting ${subfolder:+$subfolder/}$dest_folder ..."
    if [ "$force_mkdir" = "mkdir" ]; then
        mkdir -p "$dest"
        unzip -qo "$zipfile" -d "$dest"
    else
        unzip -qo "$zipfile" -d "$extract_dir"
    fi

    rm -f "$zipfile"
done

# ── Clone git repo folders (sparse checkout) ─────────────────────────────────
for entry in "${GIT_REPOS[@]}"; do
    read -r repo commit dest_folder sparse_paths <<< "$entry"
    dest="$DATA_DIR/$dest_folder"

    IFS=',' read -ra paths <<< "$sparse_paths"

    if [ -d "$dest/.git" ]; then
        echo "Updating sparse-checkout for $dest_folder ..."
        git -C "$dest" sparse-checkout set "${paths[@]}"
        git -C "$dest" fetch --depth 1 --filter=blob:none -q origin "$commit"
        git -C "$dest" checkout --detach FETCH_HEAD -q
    else
        echo "Sparse-cloning $dest_folder from $repo ..."
        mkdir -p "$dest"
        git -C "$dest" init -q
        git -C "$dest" remote add origin "$repo"
        git -C "$dest" sparse-checkout init --cone
        git -C "$dest" sparse-checkout set "${paths[@]}"
        git -C "$dest" fetch --depth 1 --filter=blob:none -q origin "$commit"
        git -C "$dest" checkout --detach FETCH_HEAD -q
    fi
done

echo "Done. Assets are in $DATA_DIR"
