"""
Music notation output formatting module.

This module converts MusicElement objects (from omr_parser.py) into multiple
output formats: MusicXML, ABC notation, MIDI, and Markdown metadata.

Pipeline Position: OMR Parser → Music Formatter → Writer (file output)
"""

import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from grounding.omr_parser import MusicElement

logger = logging.getLogger("grounding.music_formatter")


class FormattingError(Exception):
    """Raised when music notation formatting fails."""

    def __init__(self, format_type: str, message: str):
        """
        Initialize formatting error with format type and message.

        Args:
            format_type: The format that failed (e.g., "MusicXML", "ABC", "MIDI")
            message: Error description
        """
        self.format_type = format_type
        super().__init__(f"{format_type} formatting error: {message}")


def _convert_elements_to_stream(music_elements: List[MusicElement]):
    """
    Convert MusicElement list to music21 Stream object.

    Args:
        music_elements: List of MusicElement objects from OMR parser

    Returns:
        music21 Stream object containing the musical notation

    Raises:
        FormattingError: If music21 import fails or conversion fails
    """
    try:
        from music21 import stream, note, clef, key, meter, chord
    except ImportError:
        raise FormattingError(
            "music21",
            "music21 library not installed. Run: pip install music21>=9.1.0"
        )

    if not music_elements:
        raise FormattingError("Conversion", "Cannot convert empty music_elements list")

    try:
        s = stream.Stream()

        # Extract metadata from first elements
        key_sig = None
        time_sig = None
        clef_obj = None

        # First pass: collect metadata elements
        for elem in music_elements:
            if elem.element_type == "clef":
                # Default to treble clef
                clef_obj = clef.TrebleClef()
            elif elem.element_type == "key_sig":
                # Default to C major if no pitch specified
                sharps = elem.metadata.get("sharps", 0) if elem.metadata else 0
                key_sig = key.KeySignature(sharps)
            elif elem.element_type == "time_sig":
                # Default to 4/4 if no metadata
                time_sig_str = elem.metadata.get("time_signature", "4/4") if elem.metadata else "4/4"
                time_sig = meter.TimeSignature(time_sig_str)

        # Add metadata to stream
        if clef_obj:
            s.append(clef_obj)
        if key_sig:
            s.append(key_sig)
        if time_sig:
            s.append(time_sig)

        # Second pass: add notes and rests
        for elem in music_elements:
            if elem.element_type == "note" and elem.pitch and elem.duration:
                n = note.Note(elem.pitch, quarterLength=elem.duration)
                s.append(n)
            elif elem.element_type == "rest" and elem.duration:
                r = note.Rest(quarterLength=elem.duration)
                s.append(r)

        logger.debug(f"Converted {len(music_elements)} elements to music21 Stream")
        return s

    except Exception as e:
        raise FormattingError("Conversion", f"Failed to convert elements to stream: {e}")


def format_to_musicxml(music_elements: List[MusicElement]) -> str:
    """
    Convert MusicElement list to MusicXML format string.

    MusicXML is the primary output format (most semantically rich for agent analysis).

    Args:
        music_elements: List of MusicElement objects from OMR parser

    Returns:
        MusicXML string representation

    Raises:
        FormattingError: If conversion or export fails

    Example:
        >>> elements = [MusicElement(element_type="note", measure_number=1, pitch="C4", duration=1.0)]
        >>> musicxml_str = format_to_musicxml(elements)
        >>> print(musicxml_str[:39])
        <?xml version="1.0" encoding="UTF-8"?>
    """
    try:
        s = _convert_elements_to_stream(music_elements)

        # Export to MusicXML
        # music21's write() method returns a file path when writing to disk,
        # but we can use musicxml property to get the string directly
        try:
            musicxml_str = s.write('musicxml')

            # music21's write('musicxml') may return a path or bytes depending on version
            # If it returns a path, read the file
            if isinstance(musicxml_str, (str, Path)):
                with open(musicxml_str, 'r', encoding='utf-8') as f:
                    musicxml_str = f.read()
            elif isinstance(musicxml_str, bytes):
                musicxml_str = musicxml_str.decode('utf-8')

            logger.info(f"Generated MusicXML ({len(musicxml_str)} chars)")
            return musicxml_str

        except Exception as e:
            raise FormattingError("MusicXML", f"music21 export failed: {e}")

    except FormattingError:
        raise
    except Exception as e:
        raise FormattingError("MusicXML", f"Unexpected error: {e}")


def format_to_abc(music_elements: List[MusicElement]) -> str:
    """
    Convert MusicElement list to ABC notation format.

    ABC is a text-based, human-readable music notation format.

    Args:
        music_elements: List of MusicElement objects from OMR parser

    Returns:
        ABC notation string

    Raises:
        FormattingError: If conversion or export fails

    Example:
        >>> elements = [MusicElement(element_type="note", measure_number=1, pitch="C4", duration=1.0)]
        >>> abc_str = format_to_abc(elements)
        >>> print("X:1" in abc_str)
        True
    """
    if not music_elements:
        raise FormattingError("ABC", "Cannot convert empty music_elements list")

    try:
        # Generate basic ABC notation manually
        # music21's ABC export has issues, so we'll create a simple ABC manually

        # Extract metadata
        key_sig, time_sig = _extract_metadata(music_elements)
        if not key_sig:
            key_sig = "C"  # Default to C major
        else:
            # Convert "C Major" to "C"
            key_sig = key_sig.split()[0]

        if not time_sig:
            time_sig = "4/4"

        # Build ABC header
        abc_content = "X:1\n"
        abc_content += "T:Untitled\n"
        abc_content += f"M:{time_sig}\n"
        abc_content += "L:1/4\n"
        abc_content += f"K:{key_sig}\n"

        # Convert notes to ABC notation
        current_measure = None
        for elem in music_elements:
            if elem.element_type == "note" and elem.pitch and elem.duration:
                # Add barline if measure changed
                if current_measure is not None and current_measure != elem.measure_number:
                    abc_content += "|"
                current_measure = elem.measure_number

                # Convert pitch to ABC format (e.g., "C4" -> "C")
                # ABC notation: lowercase = higher octave, uppercase = middle octave
                pitch = elem.pitch[0]  # Get note letter
                octave = int(elem.pitch[1]) if len(elem.pitch) > 1 and elem.pitch[1].isdigit() else 4

                # ABC octave notation: C4 = C, C5 = c, C3 = C,
                if octave >= 5:
                    pitch = pitch.lower()
                elif octave <= 3:
                    pitch = pitch.upper() + ","

                # Add accidentals if present
                if len(elem.pitch) > 2:
                    if elem.pitch[1] == "#":
                        pitch = "^" + pitch
                    elif elem.pitch[1] == "b":
                        pitch = "_" + pitch

                # Add duration (simplified - just use quarterLength)
                if elem.duration == 0.5:
                    pitch += "/2"
                elif elem.duration == 2.0:
                    pitch += "2"

                abc_content += pitch

            elif elem.element_type == "rest" and elem.duration:
                if current_measure is not None and current_measure != elem.measure_number:
                    abc_content += "|"
                current_measure = elem.measure_number

                # Rest in ABC is 'z'
                rest = "z"
                if elem.duration == 0.5:
                    rest += "/2"
                elif elem.duration == 2.0:
                    rest += "2"

                abc_content += rest

        # Add final barline
        abc_content += "|\n"

        logger.info(f"Generated ABC notation ({len(abc_content)} chars)")
        return abc_content

    except FormattingError:
        raise
    except Exception as e:
        raise FormattingError("ABC", f"Unexpected error: {e}")


def format_to_midi(music_elements: List[MusicElement]) -> bytes:
    """
    Convert MusicElement list to MIDI file bytes.

    MIDI is for audio playback, not semantic analysis.

    Args:
        music_elements: List of MusicElement objects from OMR parser

    Returns:
        MIDI file as bytes

    Raises:
        FormattingError: If conversion or export fails

    Example:
        >>> elements = [MusicElement(element_type="note", measure_number=1, pitch="C4", duration=1.0)]
        >>> midi_bytes = format_to_midi(elements)
        >>> midi_bytes[:4]
        b'MThd'
    """
    try:
        s = _convert_elements_to_stream(music_elements)

        # Export to MIDI
        try:
            import tempfile

            # music21's write('midi') requires a file path
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                tmp_path = tmp.name

            s.write('midi', fp=tmp_path)

            # Read the MIDI file bytes
            with open(tmp_path, 'rb') as f:
                midi_bytes = f.read()

            # Clean up temp file
            Path(tmp_path).unlink()

            logger.info(f"Generated MIDI file ({len(midi_bytes)} bytes)")
            return midi_bytes

        except Exception as e:
            raise FormattingError("MIDI", f"music21 MIDI export failed: {e}")

    except FormattingError:
        raise
    except Exception as e:
        raise FormattingError("MIDI", f"Unexpected error: {e}")


def format_to_markdown(music_elements: List[MusicElement]) -> str:
    """
    Generate Markdown metadata summary for music notation.

    Extracts and formats key signature, time signature, tempo, and statistics.

    Args:
        music_elements: List of MusicElement objects from OMR parser

    Returns:
        Markdown string with music metadata section

    Raises:
        FormattingError: If metadata extraction fails

    Example:
        >>> elements = [
        ...     MusicElement(element_type="key_sig", measure_number=1, metadata={"sharps": 0}),
        ...     MusicElement(element_type="time_sig", measure_number=1, metadata={"time_signature": "4/4"}),
        ...     MusicElement(element_type="note", measure_number=1, pitch="C4", duration=1.0)
        ... ]
        >>> md = format_to_markdown(elements)
        >>> "## Music Metadata" in md
        True
    """
    if not music_elements:
        raise FormattingError("Markdown", "Cannot generate metadata from empty music_elements list")

    try:
        # Extract metadata
        key_signature = "Unknown"
        time_signature = "Unknown"
        tempo = "Unknown"

        note_count = 0
        rest_count = 0
        measures = set()

        for elem in music_elements:
            if elem.element_type == "key_sig" and elem.metadata:
                sharps = elem.metadata.get("sharps", 0)
                # Simple key detection (could be enhanced)
                key_map = {
                    0: "C Major / A minor",
                    1: "G Major / E minor",
                    -1: "F Major / D minor",
                    2: "D Major / B minor",
                    -2: "Bb Major / G minor",
                }
                key_signature = key_map.get(sharps, f"{sharps} sharps/flats")

            elif elem.element_type == "time_sig" and elem.metadata:
                time_signature = elem.metadata.get("time_signature", "Unknown")

            elif elem.element_type == "note":
                note_count += 1
                measures.add(elem.measure_number)

            elif elem.element_type == "rest":
                rest_count += 1
                measures.add(elem.measure_number)

        measure_count = len(measures) if measures else 0

        # Build markdown
        markdown = "## Music Metadata\n\n"
        markdown += f"- **Key Signature:** {key_signature}\n"
        markdown += f"- **Time Signature:** {time_signature}\n"
        markdown += f"- **Tempo:** {tempo}\n"
        markdown += f"- **Measure Count:** {measure_count}\n"
        markdown += f"- **Note Count:** {note_count}\n"
        markdown += f"- **Rest Count:** {rest_count}\n"
        markdown += f"- **Total Elements:** {len(music_elements)}\n"
        markdown += "\n"
        markdown += "### Available Formats\n\n"
        markdown += "- MusicXML: `music.musicxml`\n"
        markdown += "- ABC Notation: `music.abc`\n"
        markdown += "- MIDI: `music.mid`\n"

        logger.info(f"Generated markdown metadata ({measure_count} measures, {note_count} notes)")
        return markdown

    except Exception as e:
        raise FormattingError("Markdown", f"Failed to generate metadata: {e}")


def extract_music_metadata(music_elements: List[MusicElement]) -> dict:
    """
    Extract structured music metadata for manifest generation.

    Args:
        music_elements: List of MusicElement objects from OMR parser

    Returns:
        Dictionary with music metadata: key, time_signature, phrase_count, measure_count
    """
    if not music_elements:
        return {}

    key_sig, time_sig = _extract_metadata(music_elements)

    # Count measures
    measures = set(elem.measure_number for elem in music_elements if elem.measure_number)
    measure_count = len(measures) if measures else 0

    # Estimate phrase count (rough heuristic: 4 measures per phrase)
    phrase_count = max(1, measure_count // 4)

    metadata = {
        "measure_count": measure_count,
        "phrase_count": phrase_count,
    }

    if key_sig:
        metadata["key"] = key_sig
    if time_sig:
        metadata["time_signature"] = time_sig

    return metadata


def _extract_metadata(music_elements: List[MusicElement]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract key signature and time signature from music elements.

    Helper function for metadata extraction.

    Args:
        music_elements: List of MusicElement objects

    Returns:
        Tuple of (key_signature, time_signature) as strings or None
    """
    key_sig = None
    time_sig = None

    for elem in music_elements:
        if elem.element_type == "key_sig" and elem.metadata:
            sharps = elem.metadata.get("sharps", 0)
            key_map = {
                0: "C Major",
                1: "G Major",
                -1: "F Major",
                2: "D Major",
                -2: "Bb Major",
            }
            key_sig = key_map.get(sharps, f"{sharps} sharps")

        elif elem.element_type == "time_sig" and elem.metadata:
            time_sig = elem.metadata.get("time_signature", "4/4")

    return key_sig, time_sig
