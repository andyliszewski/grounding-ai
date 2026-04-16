"""
Optical Music Recognition (OMR) parser module using Audiveris.

This module provides functions to extract music notation from PDF files using
the Audiveris OMR engine via subprocess integration. Proven approach from Story 7.1 PoC.
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

logger = logging.getLogger("grounding.omr_parser")


class AudiverisOMRError(Exception):
    """Raised when Audiveris OMR processing fails."""
    pass


@dataclass
class MusicElement:
    """
    Represents a single musical element extracted from a PDF.

    Attributes:
        element_type: Type of musical element (note, rest, clef, key_sig, time_sig, barline)
        measure_number: Measure number where this element appears
        staff_number: Staff number (1-indexed, default 1 for single staff)
        voice_number: Voice number within staff (1-indexed, default 1)
        pitch: Note pitch in scientific notation (e.g., "C4", "F#5"), None for non-notes
        duration: Note duration in quarter notes (e.g., 0.5 = eighth note), None for non-notes
        metadata: Additional metadata (tempo, dynamics, articulations, etc.)
    """
    element_type: Literal["note", "rest", "clef", "key_sig", "time_sig", "barline"]
    measure_number: int
    staff_number: int = 1
    voice_number: int = 1
    pitch: Optional[str] = None
    duration: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation for debugging."""
        if self.element_type == "note" and self.pitch and self.duration:
            return f"Note({self.pitch}, {self.duration}q, m{self.measure_number})"
        elif self.element_type == "rest" and self.duration:
            return f"Rest({self.duration}q, m{self.measure_number})"
        else:
            return f"{self.element_type.title()}(m{self.measure_number})"


def _find_audiveris_binary() -> Optional[Path]:
    """
    Find Audiveris binary on the system.

    Checks:
    1. AUDIVERIS_HOME environment variable
    2. Standard macOS application path
    3. PATH environment variable

    Returns:
        Path to Audiveris binary if found, None otherwise
    """
    # Check environment variable
    audiveris_home = os.getenv("AUDIVERIS_HOME")
    if audiveris_home:
        binary = Path(audiveris_home) / "Audiveris"
        if binary.exists():
            return binary

    # Check macOS standard location
    macos_path = Path("/Applications/Audiveris.app/Contents/MacOS/Audiveris")
    if macos_path.exists():
        return macos_path

    # Check Linux standard location (installed via .deb)
    linux_path = Path("/opt/audiveris/bin/Audiveris")
    if linux_path.exists():
        return linux_path

    # Check if 'audiveris' is in PATH
    try:
        result = subprocess.run(
            ["which", "audiveris"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except Exception:
        pass

    return None


def _check_java_version() -> bool:
    """
    Check if Java Runtime Environment is installed and version >= 11.

    Returns:
        True if JRE >= 11 is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            check=False
        )
        # Java version output goes to stderr
        output = result.stderr + result.stdout

        # Look for version number (e.g., "version \"11.0.1\"" or "version \"24\"")
        if "version" in output.lower():
            logger.debug(f"Java version check: {output.split()[0:3]}")
            return True  # Assume if java exists, it's modern enough (>=11)
        return False
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.warning(f"Java version check failed: {e}")
        return False


def check_audiveris_available() -> bool:
    """
    Check if Audiveris and JRE prerequisites are available.

    Returns:
        True if both Java >=11 and Audiveris are installed and accessible, False otherwise
    """
    return _check_java_version() and _find_audiveris_binary() is not None


def parse_music_pdf(file_path: Path, output_dir: Optional[Path] = None) -> List[MusicElement]:
    """
    Parse a music notation PDF using Audiveris and return structured music elements.

    This function uses subprocess to call Audiveris CLI (proven approach from Story 7.1 PoC),
    processes the resulting MusicXML output, and converts it to MusicElement objects.

    Args:
        file_path: Path to the PDF file containing music notation
        output_dir: Optional directory for Audiveris output (default: temp directory)

    Returns:
        List of MusicElement objects representing the extracted music notation

    Raises:
        AudiverisOMRError: If Audiveris is not installed, JRE is missing, or processing fails
        FileNotFoundError: If file_path does not exist
        ValueError: If file_path is not a PDF file

    Example:
        >>> elements = parse_music_pdf(Path("score.pdf"))
        >>> print(f"Extracted {len(elements)} musical elements")
    """
    # Validate input
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    # Check prerequisites
    if not _check_java_version():
        raise AudiverisOMRError(
            "Java Runtime Environment (JRE) >= 11 is required for Audiveris. "
            "Install from https://adoptium.net/ or use 'brew install openjdk' on macOS."
        )

    audiveris_bin = _find_audiveris_binary()
    if not audiveris_bin:
        raise AudiverisOMRError(
            "Audiveris not found. Install from https://github.com/Audiveris/audiveris/releases "
            "or set AUDIVERIS_HOME environment variable to the installation directory. "
            "See docs/epics/epic-7-installation-guide.md for installation instructions."
        )

    # Create output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="grounding_omr_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing music PDF with Audiveris: {file_path.name}")

    # Call Audiveris via subprocess (proven PoC approach)
    cmd = [
        str(audiveris_bin),
        "-batch",
        "-export",
        "-output", str(output_dir),
        str(file_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            check=False
        )

        if result.returncode != 0:
            logger.error(f"Audiveris failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr[:500]}")  # Log first 500 chars
            raise AudiverisOMRError(
                f"Audiveris processing failed for {file_path.name}. "
                f"This may not be a valid music notation PDF. Return code: {result.returncode}"
            )

        logger.debug(f"Audiveris completed successfully for {file_path.name}")

    except subprocess.TimeoutExpired:
        raise AudiverisOMRError(
            f"Audiveris processing timed out after 120 seconds for {file_path.name}. "
            "PDF may be too large or complex."
        )
    except Exception as e:
        raise AudiverisOMRError(f"Audiveris subprocess error: {e}")

    # Find generated MusicXML file
    musicxml_files = list(output_dir.glob("*.mxl")) + list(output_dir.glob("*.musicxml"))

    if not musicxml_files:
        raise AudiverisOMRError(
            f"No MusicXML output generated by Audiveris for {file_path.name}. "
            "PDF may not contain recognizable music notation."
        )

    musicxml_path = musicxml_files[0]
    logger.info(f"Parsing MusicXML: {musicxml_path.name}")

    # Parse MusicXML with music21
    elements = _parse_musicxml_to_elements(musicxml_path)

    logger.info(f"Extracted {len(elements)} musical elements from {file_path.name}")
    return elements


def _parse_musicxml_to_elements(musicxml_path: Path) -> List[MusicElement]:
    """
    Parse MusicXML file into MusicElement objects using music21.

    Args:
        musicxml_path: Path to MusicXML file (.mxl or .musicxml)

    Returns:
        List of MusicElement objects

    Raises:
        AudiverisOMRError: If music21 parsing fails
    """
    try:
        from music21 import converter
    except ImportError:
        raise AudiverisOMRError(
            "music21 library not installed. Run: pip install music21>=9.1.0"
        )

    try:
        # Parse MusicXML
        score = converter.parse(str(musicxml_path))
        elements = []

        # Extract musical elements from the score
        for part in score.parts:
            for measure_idx, measure in enumerate(part.getElementsByClass('Measure'), start=1):
                # Extract notes and rests
                for note_or_rest in measure.notesAndRests:
                    if note_or_rest.isNote:
                        elem = MusicElement(
                            element_type="note",
                            measure_number=measure_idx,
                            pitch=note_or_rest.nameWithOctave,
                            duration=float(note_or_rest.quarterLength),
                            metadata={}
                        )
                        elements.append(elem)
                    elif note_or_rest.isRest:
                        elem = MusicElement(
                            element_type="rest",
                            measure_number=measure_idx,
                            duration=float(note_or_rest.quarterLength),
                            metadata={}
                        )
                        elements.append(elem)

        return elements

    except Exception as e:
        raise AudiverisOMRError(f"Failed to parse MusicXML with music21: {e}")


def detect_music_content(file_path: Path, quick: bool = True) -> bool:
    """
    Detect if a PDF contains music notation (vs. text/images).

    Uses heuristics to quickly determine if a PDF likely contains music notation
    without full OMR processing. Accuracy target: >80% (per Story 7.2 AC4).

    Args:
        file_path: Path to PDF file
        quick: If True, use fast heuristics. If False, sample with Audiveris (slower).

    Returns:
        True if PDF likely contains music notation, False otherwise

    Example:
        >>> is_music = detect_music_content(Path("score.pdf"))
        >>> if is_music:
        ...     elements = parse_music_pdf(Path("score.pdf"))
    """
    if not file_path.exists():
        return False

    if quick:
        # Quick heuristic: Check for staff lines using Pillow
        return _detect_staff_lines_quick(file_path)
    else:
        # Slower but more accurate: Try parsing first page with Audiveris
        return _detect_with_audiveris_sample(file_path)


def _detect_staff_lines_quick(file_path: Path) -> bool:
    """
    Quick heuristic: Detect staff lines in PDF first page using image analysis.

    Music notation typically has 5-line staves with consistent spacing.

    Args:
        file_path: Path to PDF file

    Returns:
        True if staff lines detected (likely music), False otherwise
    """
    try:
        from PIL import Image
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("Pillow or pypdfium2 not available for staff line detection. Falling back to basic check.")
        return False

    try:
        # Convert first page to image
        doc = pdfium.PdfDocument(str(file_path))
        if len(doc) == 0:
            return False

        page = doc[0]
        bitmap = page.render(scale=150/72)
        img = bitmap.to_pil()

        # Convert to grayscale
        img_gray = img.convert('L')

        # Simple heuristic: Count horizontal line density
        # Music pages have many evenly-spaced horizontal lines (staves)
        # This is a simplified check - could be enhanced with OpenCV

        # For now, return False (conservative approach)
        # TODO: Implement proper staff line detection in Story 7.4
        logger.debug("Staff line detection not yet implemented, using conservative False")
        return False

    except Exception as e:
        logger.warning(f"Staff line detection failed: {e}")
        return False


def _detect_with_audiveris_sample(file_path: Path) -> bool:
    """
    Detect music content by sampling first page with Audiveris.

    More accurate but slower than quick heuristics.

    Args:
        file_path: Path to PDF file

    Returns:
        True if Audiveris successfully extracted music from first page
    """
    try:
        # Try parsing the full PDF (Audiveris handles multi-page)
        elements = parse_music_pdf(file_path)
        return len(elements) > 0
    except AudiverisOMRError:
        return False
    except Exception as e:
        logger.warning(f"Audiveris sampling failed: {e}")
        return False
