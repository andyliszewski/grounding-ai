"""
Music description generation for semantic embedding.

This module analyzes music21 Stream objects and generates natural language descriptions
for semantic search. Part of Epic 7 Story 7.6 - Music Embeddings & Semantic Search.

Pipeline: music21 Stream → analyze features → text description → embedder → vector
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("grounding.music_descriptions")


class MusicAnalysisError(Exception):
    """Raised when music analysis fails."""
    pass


def _analyze_key_signature(stream) -> Tuple[str, str]:
    """
    Analyze the key signature of a music stream.

    Args:
        stream: music21 Stream object

    Returns:
        Tuple of (tonic_name, mode) e.g., ("C", "major")

    Raises:
        MusicAnalysisError: If key analysis fails
    """
    try:
        from music21 import key as music21_key
    except ImportError:
        raise MusicAnalysisError(
            "music21 library not installed. Run: pip install music21>=9.1.0"
        )

    try:
        # Use music21's key analysis
        key_result = stream.analyze('key')
        tonic = key_result.tonic.name
        mode = key_result.mode
        logger.debug(f"Detected key: {tonic} {mode}")
        return tonic, mode
    except Exception as e:
        logger.warning(f"Key analysis failed, defaulting to C major: {e}")
        return "C", "major"


def _get_time_signature(stream) -> str:
    """
    Extract time signature from stream.

    Args:
        stream: music21 Stream object

    Returns:
        Time signature as string (e.g., "4/4", "3/4")
    """
    try:
        time_sigs = stream.flatten().getElementsByClass('TimeSignature')
        if time_sigs:
            ts = time_sigs[0]
            return ts.ratioString
        else:
            logger.debug("No time signature found, defaulting to 4/4")
            return "4/4"
    except Exception as e:
        logger.warning(f"Time signature extraction failed, defaulting to 4/4: {e}")
        return "4/4"


def _analyze_harmony(stream, key_obj) -> List[str]:
    """
    Analyze harmonic progression using chord detection and Roman numeral analysis.

    Args:
        stream: music21 Stream object
        key_obj: music21 Key object (from stream.analyze('key'))

    Returns:
        List of Roman numeral chord symbols (e.g., ['I', 'IV', 'V', 'I'])

    Raises:
        MusicAnalysisError: If harmony analysis fails critically
    """
    try:
        from music21 import roman, chord as music21_chord
    except ImportError:
        raise MusicAnalysisError(
            "music21 library not installed. Run: pip install music21>=9.1.0"
        )

    roman_numerals = []

    try:
        # Chordify reduces polyphonic music to chord stream
        chords = stream.chordify()

        # Extract chords and convert to Roman numerals
        for c in chords.flatten().getElementsByClass('Chord'):
            try:
                # Convert chord to Roman numeral in the detected key
                rn = roman.romanNumeralFromChord(c, key_obj)
                roman_numerals.append(str(rn.figure))
            except Exception as e:
                # Skip chords that can't be analyzed (e.g., non-triadic harmonies)
                logger.debug(f"Could not analyze chord: {e}")
                continue

        logger.debug(f"Detected harmonic progression: {roman_numerals}")

        # If no chords detected, return default
        if not roman_numerals:
            logger.debug("No chords detected in harmony analysis")
            return ["I"]  # Default to tonic

        return roman_numerals

    except Exception as e:
        logger.warning(f"Harmony analysis failed: {e}")
        return ["I"]  # Default to tonic chord


def _analyze_rhythm(stream) -> str:
    """
    Analyze rhythmic patterns in the music.

    Detects:
    - Note durations (whole, half, quarter, eighth, sixteenth notes)
    - Syncopation (notes on weak beats)
    - Common rhythmic motifs (dotted rhythms, triplets)

    Args:
        stream: music21 Stream object

    Returns:
        String description of rhythmic patterns
    """
    try:
        from music21 import note as music21_note
    except ImportError:
        raise MusicAnalysisError(
            "music21 library not installed. Run: pip install music21>=9.1.0"
        )

    try:
        notes = stream.flatten().notes
        if not notes:
            return "no rhythmic content"

        # Collect note durations
        durations = []
        has_syncopation = False
        has_dotted = False
        has_triplets = False

        for n in notes:
            duration_ql = n.quarterLength  # Duration in quarter notes

            # Track duration types
            durations.append(duration_ql)

            # Detect dotted rhythms (duration is 1.5x a power of 2)
            if duration_ql in [0.75, 1.5, 3.0]:  # Dotted eighth, quarter, half
                has_dotted = True

            # Detect triplets (duration is 1/3 or 2/3)
            if abs(duration_ql - 1/3) < 0.01 or abs(duration_ql - 2/3) < 0.01:
                has_triplets = True

            # Detect syncopation (note on weak beat)
            # Simplified: check if note starts on non-integer beat
            try:
                beat = n.beat
                if beat and not beat.is_integer():
                    has_syncopation = True
            except Exception:
                pass  # Skip if beat detection fails

        # Describe duration types
        duration_desc = []
        if any(d >= 4.0 for d in durations):
            duration_desc.append("whole notes")
        if any(2.0 <= d < 4.0 for d in durations):
            duration_desc.append("half notes")
        if any(1.0 <= d < 2.0 for d in durations):
            duration_desc.append("quarter notes")
        if any(0.5 <= d < 1.0 for d in durations):
            duration_desc.append("eighth notes")
        if any(d < 0.5 for d in durations):
            duration_desc.append("sixteenth notes")

        # Build description
        rhythm_parts = []
        if duration_desc:
            rhythm_parts.append(", ".join(duration_desc[:3]))  # Limit to top 3

        if has_dotted:
            rhythm_parts.append("dotted rhythms")
        if has_triplets:
            rhythm_parts.append("triplets")
        if has_syncopation:
            rhythm_parts.append("syncopation")

        if rhythm_parts:
            rhythm_str = ", ".join(rhythm_parts)
        else:
            rhythm_str = "regular rhythmic pattern"

        logger.debug(f"Detected rhythm: {rhythm_str}")
        return rhythm_str

    except Exception as e:
        logger.warning(f"Rhythm analysis failed: {e}")
        return "regular rhythmic pattern"


def generate_music_description(stream) -> str:
    """
    Generate natural language description of music for embedding.

    Analyzes a music21 Stream and produces a text description suitable for
    semantic embedding using the all-MiniLM-L6-v2 text model.

    Args:
        stream: music21 Stream object (from music_formatter or direct parsing)

    Returns:
        Natural language description string

    Raises:
        MusicAnalysisError: If critical analysis fails or music21 unavailable
        ValueError: If stream is None or invalid

    Example:
        >>> from music21 import stream, note
        >>> s = stream.Stream()
        >>> s.append(note.Note('C4', quarterLength=1.0))
        >>> desc = generate_music_description(s)
        >>> print(desc)
        Music phrase in C major, 4/4 time signature. Harmonic progression: I. Rhythmic pattern: quarter notes. Musical characteristics: simple melody.
    """
    if stream is None:
        raise ValueError("stream cannot be None")

    try:
        from music21 import stream as music21_stream
    except ImportError:
        raise MusicAnalysisError(
            "music21 library not installed. Run: pip install music21>=9.1.0"
        )

    if not isinstance(stream, music21_stream.Stream):
        raise ValueError(f"stream must be a music21 Stream, got {type(stream).__name__}")

    logger.info("Generating music description for embedding")

    # Analyze musical features
    try:
        tonic, mode = _analyze_key_signature(stream)
        time_sig = _get_time_signature(stream)

        # Get key object for harmony analysis
        key_obj = stream.analyze('key')
        harmony = _analyze_harmony(stream, key_obj)
        rhythm = _analyze_rhythm(stream)

        # Analyze additional characteristics
        note_count = len(stream.flatten().notes)
        measure_count = len(stream.getElementsByClass('Measure'))

        # Determine musical characteristics
        characteristics = []
        if note_count < 10:
            characteristics.append("simple melody")
        elif note_count < 30:
            characteristics.append("moderate complexity")
        else:
            characteristics.append("complex passage")

        # Detect melodic patterns
        try:
            notes = stream.flatten().notes
            if len(notes) >= 4:
                pitches = [n.pitch.midi for n in notes if hasattr(n, 'pitch')]
                if len(pitches) >= 4:
                    # Check for ascending/descending scales
                    ascending = all(pitches[i] < pitches[i+1] for i in range(min(4, len(pitches)-1)))
                    descending = all(pitches[i] > pitches[i+1] for i in range(min(4, len(pitches)-1)))

                    if ascending:
                        characteristics.append("ascending melodic pattern")
                    elif descending:
                        characteristics.append("descending melodic pattern")
        except Exception:
            pass  # Skip melodic pattern detection if it fails

        # Build description using template
        harmony_str = " - ".join(harmony[:8])  # Limit to first 8 chords to avoid too long descriptions

        description = (
            f"Music phrase in {tonic} {mode}, {time_sig} time signature. "
            f"Harmonic progression: {harmony_str}. "
            f"Rhythmic pattern: {rhythm}. "
            f"Musical characteristics: {', '.join(characteristics)}."
        )

        logger.info(f"Generated description ({len(description)} chars)")
        logger.debug(f"Description: {description[:100]}...")

        return description

    except MusicAnalysisError:
        raise
    except Exception as e:
        error_msg = f"Failed to generate music description: {e}"
        logger.error(error_msg, exc_info=True)
        raise MusicAnalysisError(error_msg) from e


def generate_description_from_musicxml(musicxml_path: str) -> str:
    """
    Convenience function to generate description directly from MusicXML file.

    Args:
        musicxml_path: Path to MusicXML file

    Returns:
        Natural language description string

    Raises:
        MusicAnalysisError: If parsing or analysis fails
        FileNotFoundError: If MusicXML file not found
    """
    try:
        from music21 import converter
    except ImportError:
        raise MusicAnalysisError(
            "music21 library not installed. Run: pip install music21>=9.1.0"
        )

    try:
        logger.info(f"Parsing MusicXML file: {musicxml_path}")
        stream = converter.parse(musicxml_path)
        return generate_music_description(stream)
    except FileNotFoundError:
        raise
    except Exception as e:
        error_msg = f"Failed to parse MusicXML file {musicxml_path}: {e}"
        logger.error(error_msg, exc_info=True)
        raise MusicAnalysisError(error_msg) from e
