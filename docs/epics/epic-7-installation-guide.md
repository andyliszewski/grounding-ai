# Epic 7: OMR Support Installation Guide

**Created:** 2025-10-18
**Epic:** Epic 7 - Optical Music Recognition (OMR) Support
**Status:** Draft (Story 7.1)

---

## Overview

This guide provides step-by-step installation instructions for all components needed to run Epic 7 (OMR Support) in pdf2llm. The installation process includes:

1. **Java Runtime Environment (JRE 24+)** - Required to run Audiveris
2. **Audiveris OMR Engine** - Primary music notation recognition tool
3. **Python Dependencies** - music21, py4j (optional)
4. **Test Data** - Sample music notation PDFs for validation

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | macOS 10.14+, Linux (Ubuntu 20.04+), Windows 10+ |
| **Python** | 3.10 or higher |
| **Java** | JDK 24 or higher (JRE included) |
| **Disk Space** | ~700MB (200MB Audiveris + 500MB JDK) |
| **RAM** | 4GB minimum, 8GB recommended |
| **Processor** | x86_64 or ARM64 (Apple Silicon supported) |

### Supported Platforms

- ✅ macOS (Intel and Apple Silicon)
- ✅ Linux (Debian-based, Fedora, Arch)
- ✅ Windows 10/11

---

## Installation: macOS (Primary Development Environment)

### Step 1: Install Java Development Kit (JDK 24+)

**Important:** Modern Java (11+) does not provide separate JRE. Install JDK which includes the runtime environment.

#### Option A: Homebrew (Recommended)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenJDK (latest version)
brew install openjdk

# Or install specific version (if needed)
# brew install openjdk@24

# Link JDK for macOS /usr/bin/java wrapper to find it
sudo ln -sfn /usr/local/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk

# Verify installation
java -version
```

**Expected output:**
```
openjdk version "24" 2025-03-18
OpenJDK Runtime Environment (build 24+...)
OpenJDK 64-Bit Server VM (build 24+...)
```

#### Option B: Temurin Distribution (Alternative)

```bash
# Install Temurin JDK (Eclipse Adoptium)
brew install --cask temurin

# Verify
java -version
```

#### Option C: Oracle JDK (Manual Download)

1. Download Oracle JDK 24 from: https://www.oracle.com/java/technologies/downloads/
2. Open `.dmg` file and follow installer
3. JDK installed to: `/Library/Java/JavaVirtualMachines/jdk-24.jdk`
4. Verify: `java -version`

### Step 2: Install Audiveris

#### Option A: DMG Installer (Recommended)

1. Visit Audiveris releases: https://github.com/Audiveris/audiveris/releases
2. Download latest `.dmg` file (e.g., `Audiveris-5.7.1.dmg`)
3. Open `.dmg` file
4. Drag **Audiveris.app** to `/Applications`
5. **IMPORTANT:** First launch requires right-click → Open (to bypass Gatekeeper)

**Verify installation:**
```bash
# Add Audiveris to PATH (optional, for CLI usage)
export PATH="/Applications/Audiveris.app/Contents/MacOS:$PATH"

# Test command-line interface
/Applications/Audiveris.app/Contents/MacOS/Audiveris -version
```

**Expected output:**
```
Audiveris Version 5.7.1
```

#### Option B: Build from Source (Advanced)

```bash
# Clone repository
git clone https://github.com/Audiveris/audiveris.git
cd audiveris

# Checkout master branch (stable releases)
git checkout master

# Build with Gradle (requires JDK 24+)
./gradlew build

# Run
./gradlew run
```

### Step 3: Install Python Dependencies

```bash
# Ensure pdf2llm virtual environment is active
cd ~/grounding-ai
source pdf2llmenv/bin/activate  # Or: ./pdf2llmenv/bin/activate

# Install music21 (BSD licensed)
pip install music21>=9.1.0

# Install py4j (optional - only if API integration needed in Story 7.2)
pip install py4j>=0.10.9
```

**Verify music21:**
```bash
python -c "import music21; print(music21.VERSION)"
```

**Expected:** `9.1.0` or higher

### Step 4: Test Installation (Basic Smoke Test)

```bash
# Create test directory
mkdir -p ~/omr_test
cd ~/omr_test

# Download a simple test score (public domain)
# For now, we'll create a simple test in Story 7.1 PoC

# Test Audiveris CLI
/Applications/Audiveris.app/Contents/MacOS/Audiveris --help
```

**Expected:** Help text showing Audiveris CLI options

---

## Installation: Linux (Ubuntu/Debian)

### Step 1: Install Java Development Kit (JDK 24+)

```bash
# Update package list
sudo apt update

# Install OpenJDK 24 (if available) or latest
sudo apt install -y openjdk-24-jdk

# Or install latest available version
sudo apt install -y default-jdk

# Verify installation
java -version
```

**If JDK 24 not available via apt:**
```bash
# Download Oracle JDK 24 manually
wget https://download.oracle.com/java/24/latest/jdk-24_linux-x64_bin.tar.gz

# Extract
sudo tar -xvf jdk-24_linux-x64_bin.tar.gz -C /opt

# Set JAVA_HOME
echo 'export JAVA_HOME=/opt/jdk-24' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
java -version
```

### Step 2: Install Audiveris

#### Option A: DEB Package (Debian/Ubuntu)

```bash
# Download .deb from GitHub releases
wget https://github.com/Audiveris/audiveris/releases/download/5.7.1/Audiveris-5.7.1.deb

# Install
sudo dpkg -i Audiveris-5.7.1.deb

# Fix dependencies if needed
sudo apt-get install -f

# Verify
audiveris -version
```

#### Option B: Flatpak (Universal Linux)

```bash
# Install Flatpak (if not already installed)
sudo apt install flatpak

# Add Flathub repository
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo

# Install Audiveris
flatpak install flathub org.audiveris.audiveris

# Verify
flatpak run org.audiveris.audiveris --version
```

### Step 3: Install Python Dependencies

```bash
# Activate virtual environment
cd /path/to/pdf2llm
source pdf2llmenv/bin/activate

# Install music21
pip install music21>=9.1.0

# Install py4j (optional)
pip install py4j>=0.10.9
```

---

## Installation: Windows

### Step 1: Install Java Development Kit (JDK 24+)

#### Option A: Oracle JDK Installer

1. Download Oracle JDK 24 from: https://www.oracle.com/java/technologies/downloads/#jdk24-windows
2. Download Windows x64 Installer (`.msi`)
3. Run installer and follow prompts
4. Add to PATH (installer should do this automatically)

**Verify in Command Prompt:**
```cmd
java -version
```

#### Option B: OpenJDK via Chocolatey

```cmd
# Install Chocolatey (if not installed)
# Visit: https://chocolatey.org/install

# Install OpenJDK
choco install openjdk --version=24

# Verify
java -version
```

### Step 2: Install Audiveris

#### MSI Installer

1. Visit: https://github.com/Audiveris/audiveris/releases
2. Download `.msi` file (e.g., `Audiveris-5.7.1.msi`)
3. Run installer
4. Follow installation wizard
5. Default install location: `C:\Program Files\Audiveris`

**Verify in Command Prompt:**
```cmd
"C:\Program Files\Audiveris\bin\Audiveris.exe" -version
```

**Add to PATH (optional):**
1. System Properties → Environment Variables
2. Edit PATH variable
3. Add: `C:\Program Files\Audiveris\bin`
4. Restart terminal

### Step 3: Install Python Dependencies

```cmd
# Activate virtual environment
cd C:\path\to\pdf2llm
pdf2llmenv\Scripts\activate

# Install music21
pip install music21>=9.1.0

# Install py4j (optional)
pip install py4j>=0.10.9
```

---

## Disk Space Requirements

| Component | Size | Notes |
|-----------|------|-------|
| **JDK 24** | ~500MB | Includes JRE |
| **Audiveris** | ~200MB | OMR engine + GUI |
| **music21** | ~50MB | Python library + corpus |
| **py4j** | ~5MB | Python-Java bridge |
| **Test PDFs** | ~10MB | Sample music notation |
| **Total** | **~765MB** | Approximate |

---

## Troubleshooting

### Issue: `java: command not found`

**Cause:** Java not in PATH

**Solution (macOS/Linux):**
```bash
# Find Java installation
/usr/libexec/java_home -V  # macOS
update-alternatives --config java  # Linux

# Add to PATH
export JAVA_HOME=$(/usr/libexec/java_home)  # macOS
export JAVA_HOME=/opt/jdk-24  # Linux
export PATH=$JAVA_HOME/bin:$PATH

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export JAVA_HOME=$(/usr/libexec/java_home)' >> ~/.zshrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc
```

**Solution (Windows):**
1. System Properties → Environment Variables
2. Add `JAVA_HOME` variable: `C:\Program Files\Java\jdk-24`
3. Edit PATH: Add `%JAVA_HOME%\bin`

### Issue: `Audiveris: command not found`

**Cause:** Audiveris not in PATH or incorrect installation

**Solution (macOS):**
```bash
# Use full path
/Applications/Audiveris.app/Contents/MacOS/Audiveris

# Or create alias
alias audiveris="/Applications/Audiveris.app/Contents/MacOS/Audiveris"

# Add to ~/.zshrc for persistence
echo 'alias audiveris="/Applications/Audiveris.app/Contents/MacOS/Audiveris"' >> ~/.zshrc
```

**Solution (Linux):**
```bash
# If installed via .deb, should be in PATH automatically
which audiveris

# If Flatpak:
flatpak run org.audiveris.audiveris
```

### Issue: macOS Gatekeeper blocks Audiveris

**Cause:** Unsigned application warning

**Solution:**
1. Right-click `Audiveris.app`
2. Select "Open"
3. Click "Open" in security dialog
4. Enter admin password if prompted
5. Subsequent launches will work normally

### Issue: `music21` import fails

**Cause:** Not installed in correct virtual environment

**Solution:**
```bash
# Ensure virtual environment is active
which python  # Should show pdf2llmenv path

# Re-install
pip uninstall music21
pip install music21>=9.1.0

# Verify
python -c "import music21; print(music21.VERSION)"
```

### Issue: JDK version mismatch

**Cause:** Audiveris requires JDK 24+, older version installed

**Solution:**
```bash
# Check current version
java -version

# If < 24, upgrade
brew upgrade openjdk  # macOS
sudo apt upgrade openjdk-24-jdk  # Linux

# Or install manually from Oracle
```

---

## Verification Checklist

After installation, verify all components:

```bash
# 1. Check Java version (should be 24+)
java -version

# 2. Check Audiveris (macOS example)
/Applications/Audiveris.app/Contents/MacOS/Audiveris -version

# 3. Check Python environment
which python  # Should show pdf2llmenv

# 4. Check music21
python -c "import music21; print(music21.VERSION)"

# 5. Check py4j (if installed)
python -c "import py4j; print(py4j.__version__)"
```

**Expected Output Summary:**
- ✅ Java version: `24` or higher
- ✅ Audiveris version: `5.7.1` or higher
- ✅ Python: Points to pdf2llmenv
- ✅ music21: `9.1.0` or higher
- ✅ py4j: `0.10.9` or higher (if installed)

---

## Next Steps

After successful installation:

1. **Story 7.1 PoC:** Test Audiveris with sample music notation PDF
2. **Story 7.2:** Implement Python integration (subprocess)
3. **Story 7.3:** Integrate MusicXML output formatting
4. **Story 7.4:** Implement hybrid document processing
5. **Story 7.5:** Add CLI integration
6. **Story 7.6:** Add music embeddings for semantic search

---

## Updating Audiveris

To update to a newer version:

### macOS
```bash
# Download new .dmg from GitHub releases
# Replace old Audiveris.app in /Applications
```

### Linux (DEB)
```bash
# Download new .deb
wget https://github.com/Audiveris/audiveris/releases/download/<VERSION>/Audiveris-<VERSION>.deb
sudo dpkg -i Audiveris-<VERSION>.deb
```

### Linux (Flatpak)
```bash
flatpak update org.audiveris.audiveris
```

### Windows
```bash
# Download and run new .msi installer
# Will upgrade existing installation
```

---

## Uninstalling

### macOS
```bash
# Remove Audiveris
rm -rf /Applications/Audiveris.app

# Remove JDK (optional, if no other Java apps)
sudo rm -rf /Library/Java/JavaVirtualMachines/jdk-24.jdk

# Uninstall via Homebrew (if installed that way)
brew uninstall openjdk
```

### Linux (DEB)
```bash
sudo apt remove audiveris
```

### Linux (Flatpak)
```bash
flatpak uninstall org.audiveris.audiveris
```

### Windows
1. Control Panel → Programs and Features
2. Select "Audiveris"
3. Click "Uninstall"

---

## Alternative: homr (Fallback Option)

If Audiveris installation proves problematic, **homr** (MIT licensed, Python-native) can be used:

```bash
# Activate virtual environment
source pdf2llmenv/bin/activate

# Install homr
pip install homr

# Verify
python -c "import homr; print('homr installed')"
```

**Pros:**
- No Java dependency
- Easy pip installation
- MIT license (permissive)

**Cons:**
- Lower accuracy (75-80% vs 85-90%)
- Smaller community

See comparison matrix for details: `docs/epics/epic-7-omr-library-comparison.md`

---

## References

- **Audiveris Handbook:** https://audiveris.github.io/audiveris/_pages/handbook/
- **Audiveris Releases:** https://github.com/Audiveris/audiveris/releases
- **Java Downloads:** https://www.oracle.com/java/technologies/downloads/
- **Homebrew:** https://brew.sh/
- **Flatpak:** https://flatpak.org/
- **music21:** https://github.com/cuthbertLab/music21
- **py4j:** https://www.py4j.org/

---

**Document Version:** 1.0
**Last Updated:** 2025-10-18
**Tested On:** macOS 14.x (Sonoma)
**Next Review:** After Story 7.1 PoC completion
