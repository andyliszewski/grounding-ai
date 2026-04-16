#!/bin/bash
# Download public domain philosophy texts for the philosopher agent corpus
# Texts organized by collection tags for agent filtering

set -e

STAGING_DIR="${STAGING_DIR:-$HOME/Documents/staging}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Philosophy Corpus Downloader ==="
echo "Staging directory: $STAGING_DIR"
echo ""

# Create collection directories
mkdir -p "$STAGING_DIR/eastern-philosophy"
mkdir -p "$STAGING_DIR/western-philosophy"
mkdir -p "$STAGING_DIR/buddhism"
mkdir -p "$STAGING_DIR/taoism"
mkdir -p "$STAGING_DIR/stoicism"
mkdir -p "$STAGING_DIR/ethics"
mkdir -p "$STAGING_DIR/consciousness"

# Function to download from Project Gutenberg
download_gutenberg() {
    local id="$1"
    local filename="$2"
    local collection="$3"
    local url="https://www.gutenberg.org/ebooks/${id}.epub.noimages"
    local dest="$STAGING_DIR/$collection/$filename.epub"

    if [[ -f "$dest" ]]; then
        echo "  [SKIP] $filename already exists"
        return 0
    fi

    echo "  [DOWN] $filename (Gutenberg #$id)"
    if curl -sL -o "$dest" "$url" 2>/dev/null; then
        # Verify it's a valid EPUB (should start with PK for zip)
        if head -c 2 "$dest" | grep -q "PK"; then
            echo "  [OK]   $filename"
        else
            echo "  [FAIL] $filename - invalid EPUB, trying TXT"
            rm -f "$dest"
            # Try plain text instead
            local txt_url="https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
            local txt_dest="$STAGING_DIR/$collection/$filename.txt"
            if curl -sL -o "$txt_dest" "$txt_url" 2>/dev/null; then
                echo "  [OK]   $filename (as TXT)"
            else
                echo "  [FAIL] $filename - could not download"
                rm -f "$txt_dest"
            fi
        fi
    else
        echo "  [FAIL] $filename"
    fi
}

# Function to download from Internet Archive
download_archive() {
    local id="$1"
    local filename="$2"
    local collection="$3"
    local format="${4:-pdf}"
    local url="https://archive.org/download/${id}/${id}.${format}"
    local dest="$STAGING_DIR/$collection/$filename.${format}"

    if [[ -f "$dest" ]]; then
        echo "  [SKIP] $filename already exists"
        return 0
    fi

    echo "  [DOWN] $filename (Archive: $id)"
    if curl -sL -o "$dest" "$url" 2>/dev/null && [[ -s "$dest" ]]; then
        echo "  [OK]   $filename"
    else
        echo "  [FAIL] $filename - trying alternate formats"
        rm -f "$dest"
        # Try to get the PDF directly from the item page
        for ext in pdf epub txt; do
            local alt_dest="$STAGING_DIR/$collection/$filename.${ext}"
            local alt_url="https://archive.org/download/${id}/${id}.${ext}"
            if curl -sL -o "$alt_dest" "$alt_url" 2>/dev/null && [[ -s "$alt_dest" ]]; then
                echo "  [OK]   $filename (as ${ext^^})"
                return 0
            fi
            rm -f "$alt_dest"
        done
        echo "  [FAIL] $filename - no format available"
    fi
}

echo ""
echo "=== TAOISM ==="
download_gutenberg 216 "tao-te-ching-legge" "taoism"
download_gutenberg 49965 "tao-te-ching-minimalist" "taoism"
download_gutenberg 59709 "chuang-tzu-mystic-moralist" "taoism"

echo ""
echo "=== CONFUCIANISM / EASTERN ==="
download_gutenberg 3330 "analects-confucius" "eastern-philosophy"
download_gutenberg 4094 "chinese-classics-vol1-analects" "eastern-philosophy"
download_gutenberg 10056 "chinese-literature-analects-mencius" "eastern-philosophy"

echo ""
echo "=== HINDUISM / VEDANTA ==="
download_gutenberg 3283 "upanishads-paramananda" "eastern-philosophy"
download_gutenberg 2388 "bhagavad-gita-arnold" "eastern-philosophy"

echo ""
echo "=== BUDDHISM ==="
download_gutenberg 2017 "dhammapada-muller" "buddhism"
download_gutenberg 35185 "dhammapada-woodward" "buddhism"
download_gutenberg 64623 "diamond-sutra-gemmell" "buddhism"

echo ""
echo "=== ANCIENT GREEK - PLATO ==="
download_gutenberg 1497 "plato-republic" "western-philosophy"
download_gutenberg 1600 "plato-symposium" "western-philosophy"
download_gutenberg 13726 "plato-apology-crito-phaedo" "western-philosophy"
download_gutenberg 1656 "plato-apology" "western-philosophy"
download_gutenberg 1658 "plato-phaedrus" "western-philosophy"
download_gutenberg 1636 "plato-meno" "western-philosophy"
download_gutenberg 1726 "plato-cratylus" "western-philosophy"

echo ""
echo "=== ANCIENT GREEK - ARISTOTLE ==="
download_gutenberg 8438 "aristotle-nicomachean-ethics" "ethics"
download_gutenberg 6762 "aristotle-politics" "western-philosophy"
download_gutenberg 1974 "aristotle-poetics" "western-philosophy"

echo ""
echo "=== PRESOCRATICS ==="
download_gutenberg 51548 "nietzsche-early-greek-philosophy" "western-philosophy"

echo ""
echo "=== STOICISM ==="
download_gutenberg 2680 "marcus-aurelius-meditations" "stoicism"
download_gutenberg 55317 "marcus-aurelius-meditations-foulis" "stoicism"
download_gutenberg 45109 "epictetus-enchiridion" "stoicism"
download_gutenberg 10661 "epictetus-discourses-enchiridion" "stoicism"
download_gutenberg 3794 "seneca-on-benefits" "stoicism"
download_gutenberg 59025 "seneca-works-index" "stoicism"

echo ""
echo "=== NEOPLATONISM ==="
download_gutenberg 42930 "plotinus-enneads-vol1" "western-philosophy"
download_gutenberg 42931 "plotinus-enneads-vol2" "western-philosophy"
download_gutenberg 42932 "plotinus-enneads-vol3" "western-philosophy"
download_gutenberg 42933 "plotinus-enneads-vol4" "western-philosophy"

echo ""
echo "=== RENAISSANCE ==="
download_gutenberg 3600 "montaigne-essays-complete" "western-philosophy"

echo ""
echo "=== MODERN - RATIONALISTS ==="
download_gutenberg 70091 "descartes-meditations" "western-philosophy"
download_gutenberg 59 "descartes-discourse-method" "western-philosophy"
download_gutenberg 3800 "spinoza-ethics" "ethics"

echo ""
echo "=== MODERN - EMPIRICISTS ==="
download_gutenberg 9662 "hume-enquiry-understanding" "western-philosophy"
download_gutenberg 4320 "hume-enquiry-morals" "ethics"

echo ""
echo "=== MODERN - KANT ==="
download_gutenberg 4280 "kant-critique-pure-reason" "western-philosophy"
download_gutenberg 52821 "kant-prolegomena" "western-philosophy"
download_gutenberg 5683 "kant-critique-practical-reason" "ethics"
download_gutenberg 5684 "kant-metaphysics-morals" "ethics"

echo ""
echo "=== EXISTENTIALISM - NIETZSCHE ==="
download_gutenberg 1998 "nietzsche-thus-spoke-zarathustra" "western-philosophy"
download_gutenberg 4363 "nietzsche-beyond-good-evil" "ethics"
download_gutenberg 52319 "nietzsche-genealogy-morals" "ethics"
download_gutenberg 7205 "nietzsche-ecce-homo" "western-philosophy"
download_gutenberg 51356 "nietzsche-antichrist" "western-philosophy"

echo ""
echo "=== EXISTENTIALISM - KIERKEGAARD ==="
download_gutenberg 60333 "kierkegaard-selections" "western-philosophy"

echo ""
echo "=== PRAGMATISM / RELIGIOUS ==="
download_gutenberg 621 "william-james-varieties-religious" "consciousness"

echo ""
echo "=== Download complete ==="
echo ""

# Count results
total=0
for dir in "$STAGING_DIR"/{eastern-philosophy,western-philosophy,buddhism,taoism,stoicism,ethics,consciousness}; do
    if [[ -d "$dir" ]]; then
        count=$(find "$dir" -type f \( -name "*.epub" -o -name "*.txt" -o -name "*.pdf" \) 2>/dev/null | wc -l)
        total=$((total + count))
        echo "  $(basename "$dir"): $count files"
    fi
done
echo ""
echo "Total files downloaded: $total"
echo ""
echo "Files are in staging directory ready for ingestion."
echo "The staging watcher will process them automatically if running."
