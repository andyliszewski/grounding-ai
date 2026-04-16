"""
Unit tests for grounding.music_formatter module.

Tests music notation output formatting functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from grounding.omr_parser import MusicElement
from grounding.music_formatter import (
    FormattingError,
    format_to_musicxml,
    format_to_abc,
    format_to_midi,
    format_to_markdown,
    _convert_elements_to_stream,
)


# Fixtures


@pytest.fixture
def simple_melody():
    """Create a simple melody with 4 notes for testing."""
    return [
        MusicElement(element_type="clef", measure_number=1),
        MusicElement(element_type="key_sig", measure_number=1, metadata={"sharps": 0}),
        MusicElement(element_type="time_sig", measure_number=1, metadata={"time_signature": "4/4"}),
        MusicElement(element_type="note", measure_number=1, pitch="C4", duration=1.0),
        MusicElement(element_type="note", measure_number=1, pitch="D4", duration=1.0),
        MusicElement(element_type="note", measure_number=1, pitch="E4", duration=1.0),
        MusicElement(element_type="note", measure_number=1, pitch="F4", duration=1.0),
    ]


@pytest.fixture
def melody_with_rests():
    """Create a melody with notes and rests."""
    return [
        MusicElement(element_type="note", measure_number=1, pitch="C4", duration=1.0),
        MusicElement(element_type="rest", measure_number=1, duration=1.0),
        MusicElement(element_type="note", measure_number=2, pitch="E4", duration=0.5),
        MusicElement(element_type="rest", measure_number=2, duration=0.5),
    ]


# FormattingError Tests


def test_formatting_error_creation():
    """Test FormattingError exception creation."""
    error = FormattingError("MusicXML", "Test error message")

    assert error.format_type == "MusicXML"
    assert "MusicXML formatting error: Test error message" in str(error)


# format_to_musicxml() Tests


def test_format_to_musicxml_simple_melody(simple_melody):
    """Test MusicXML generation from simple melody."""
    musicxml_str = format_to_musicxml(simple_melody)

    # Verify it's XML
    assert musicxml_str.startswith("<?xml")
    assert "encoding" in musicxml_str

    # Verify it contains musical content
    assert "score-partwise" in musicxml_str or "part" in musicxml_str


def test_format_to_musicxml_with_rests(melody_with_rests):
    """Test MusicXML generation with notes and rests."""
    musicxml_str = format_to_musicxml(melody_with_rests)

    assert musicxml_str.startswith("<?xml")
    # Should contain both notes and rests
    assert len(musicxml_str) > 100


def test_format_to_musicxml_empty_elements():
    """Test MusicXML generation with empty element list."""
    with pytest.raises(FormattingError) as exc_info:
        format_to_musicxml([])

    assert "empty" in str(exc_info.value).lower()


# format_to_abc() Tests


def test_format_to_abc_simple_melody(simple_melody):
    """Test ABC notation generation from simple melody."""
    abc_str = format_to_abc(simple_melody)

    # ABC format always starts with X: field
    assert "X:" in abc_str

    # Should contain ABC header fields
    # Note: music21's ABC export may vary, so we check for common patterns
    assert isinstance(abc_str, str)
    assert len(abc_str) > 10


def test_format_to_abc_with_rests(melody_with_rests):
    """Test ABC notation generation with notes and rests."""
    abc_str = format_to_abc(melody_with_rests)

    assert "X:" in abc_str
    assert isinstance(abc_str, str)


def test_format_to_abc_empty_elements():
    """Test ABC generation with empty element list."""
    with pytest.raises(FormattingError) as exc_info:
        format_to_abc([])

    assert "empty" in str(exc_info.value).lower()


# format_to_midi() Tests


def test_format_to_midi_simple_melody(simple_melody):
    """Test MIDI generation from simple melody."""
    midi_bytes = format_to_midi(simple_melody)

    # MIDI files start with "MThd" header
    assert midi_bytes[:4] == b'MThd'
    assert isinstance(midi_bytes, bytes)
    assert len(midi_bytes) > 20  # MIDI files have minimum size


def test_format_to_midi_with_rests(melody_with_rests):
    """Test MIDI generation with notes and rests."""
    midi_bytes = format_to_midi(melody_with_rests)

    assert midi_bytes[:4] == b'MThd'
    assert isinstance(midi_bytes, bytes)


def test_format_to_midi_empty_elements():
    """Test MIDI generation with empty element list."""
    with pytest.raises(FormattingError) as exc_info:
        format_to_midi([])

    assert "empty" in str(exc_info.value).lower()


def test_format_to_midi_validation_with_mido(simple_melody):
    """Test MIDI file can be parsed by mido library."""
    midi_bytes = format_to_midi(simple_melody)

    try:
        import mido
        from io import BytesIO

        # Parse MIDI bytes with mido
        midi_file = mido.MidiFile(file=BytesIO(midi_bytes))

        # Verify it has tracks
        assert len(midi_file.tracks) > 0

        # Count note_on events
        note_events = [msg for track in midi_file.tracks for msg in track if msg.type == 'note_on']
        assert len(note_events) > 0

    except ImportError:
        pytest.skip("mido not installed")


# format_to_markdown() Tests


def test_format_to_markdown_simple_melody(simple_melody):
    """Test markdown metadata generation."""
    markdown = format_to_markdown(simple_melody)

    # Should have metadata header
    assert "## Music Metadata" in markdown

    # Should have key signature
    assert "Key Signature" in markdown
    assert "C Major" in markdown  # 0 sharps = C Major

    # Should have time signature
    assert "Time Signature" in markdown
    assert "4/4" in markdown

    # Should have statistics
    assert "Note Count" in markdown
    assert "4" in markdown  # 4 notes in simple_melody


def test_format_to_markdown_with_rests(melody_with_rests):
    """Test markdown with notes and rests."""
    markdown = format_to_markdown(melody_with_rests)

    assert "## Music Metadata" in markdown
    assert "Note Count" in markdown
    assert "Rest Count" in markdown
    assert "Measure Count" in markdown


def test_format_to_markdown_empty_elements():
    """Test markdown generation with empty element list."""
    with pytest.raises(FormattingError) as exc_info:
        format_to_markdown([])

    assert "empty" in str(exc_info.value).lower()


def test_format_to_markdown_includes_format_links(simple_melody):
    """Test markdown includes links to output formats."""
    markdown = format_to_markdown(simple_melody)

    assert "MusicXML" in markdown
    assert "ABC Notation" in markdown
    assert "MIDI" in markdown
    assert "music.musicxml" in markdown
    assert "music.abc" in markdown
    assert "music.mid" in markdown


# _convert_elements_to_stream() Tests


def test_convert_elements_to_stream_simple(simple_melody):
    """Test conversion from MusicElement list to music21 Stream."""
    stream = _convert_elements_to_stream(simple_melody)

    # Should be a music21 Stream object
    from music21 import stream as m21_stream
    assert isinstance(stream, m21_stream.Stream)

    # Should have notes
    notes = stream.flatten().notes
    assert len(notes) == 4  # 4 notes in simple_melody


def test_convert_elements_to_stream_with_metadata(simple_melody):
    """Test stream contains metadata (clef, key, time signature)."""
    stream = _convert_elements_to_stream(simple_melody)

    from music21 import clef, key, meter

    # Check for clef
    clefs = stream.flatten().getElementsByClass(clef.Clef)
    assert len(clefs) > 0

    # Check for key signature
    keys = stream.flatten().getElementsByClass(key.KeySignature)
    assert len(keys) > 0

    # Check for time signature
    times = stream.flatten().getElementsByClass(meter.TimeSignature)
    assert len(times) > 0


def test_convert_elements_to_stream_empty_list():
    """Test conversion with empty element list."""
    with pytest.raises(FormattingError) as exc_info:
        _convert_elements_to_stream([])

    assert "empty" in str(exc_info.value).lower()


# Error Handling Tests


def test_format_to_musicxml_handles_conversion_error():
    """Test MusicXML generation handles conversion errors gracefully."""
    # Create invalid elements (notes without pitch)
    invalid_elements = [
        MusicElement(element_type="note", measure_number=1, pitch=None, duration=1.0)
    ]

    # Should not raise exception, should skip invalid notes
    musicxml_str = format_to_musicxml(invalid_elements)
    assert isinstance(musicxml_str, str)


def test_format_to_abc_handles_conversion_error():
    """Test ABC generation handles conversion errors gracefully."""
    # Create minimal valid element to avoid empty list error
    invalid_elements = [
        MusicElement(element_type="note", measure_number=1, pitch=None, duration=1.0)
    ]

    # Should not raise exception
    abc_str = format_to_abc(invalid_elements)
    assert isinstance(abc_str, str)


def test_format_to_midi_handles_conversion_error():
    """Test MIDI generation handles conversion errors gracefully."""
    # Create minimal valid element
    invalid_elements = [
        MusicElement(element_type="note", measure_number=1, pitch=None, duration=1.0)
    ]

    # Should not raise exception
    midi_bytes = format_to_midi(invalid_elements)
    assert isinstance(midi_bytes, bytes)


# Integration-like Tests


def test_all_formats_from_same_input(simple_melody):
    """Test generating all formats from the same input."""
    # Generate all formats
    musicxml = format_to_musicxml(simple_melody)
    abc = format_to_abc(simple_melody)
    midi = format_to_midi(simple_melody)
    markdown = format_to_markdown(simple_melody)

    # Verify all succeeded
    assert isinstance(musicxml, str)
    assert isinstance(abc, str)
    assert isinstance(midi, bytes)
    assert isinstance(markdown, str)

    # Verify content
    assert len(musicxml) > 100
    assert len(abc) > 10
    assert len(midi) > 20
    assert len(markdown) > 50


def test_format_pipeline_consistency(simple_melody):
    """Test that repeated formatting produces consistent results."""
    # Generate MusicXML twice
    musicxml1 = format_to_musicxml(simple_melody)
    musicxml2 = format_to_musicxml(simple_melody)

    # Should be identical (deterministic)
    # Note: music21 may add timestamps, so we check structure
    assert len(musicxml1) == len(musicxml2)
    assert musicxml1[:100] == musicxml2[:100]


# Edge Case Tests


def test_format_with_only_metadata_elements():
    """Test formatting with only metadata elements (no notes)."""
    metadata_only = [
        MusicElement(element_type="clef", measure_number=1),
        MusicElement(element_type="key_sig", measure_number=1, metadata={"sharps": 0}),
        MusicElement(element_type="time_sig", measure_number=1, metadata={"time_signature": "3/4"}),
    ]

    # Should generate valid output even without notes
    musicxml = format_to_musicxml(metadata_only)
    markdown = format_to_markdown(metadata_only)

    assert isinstance(musicxml, str)
    assert isinstance(markdown, str)
    assert "Note Count" in markdown and "0" in markdown


def test_format_with_different_key_signatures():
    """Test formatting with different key signatures."""
    # G Major (1 sharp)
    g_major = [
        MusicElement(element_type="key_sig", measure_number=1, metadata={"sharps": 1}),
        MusicElement(element_type="note", measure_number=1, pitch="G4", duration=1.0),
    ]

    markdown = format_to_markdown(g_major)
    assert "G Major" in markdown

    # F Major (1 flat)
    f_major = [
        MusicElement(element_type="key_sig", measure_number=1, metadata={"sharps": -1}),
        MusicElement(element_type="note", measure_number=1, pitch="F4", duration=1.0),
    ]

    markdown = format_to_markdown(f_major)
    assert "F Major" in markdown


def test_format_multi_measure_melody():
    """Test formatting melody spanning multiple measures."""
    multi_measure = [
        MusicElement(element_type="note", measure_number=1, pitch="C4", duration=1.0),
        MusicElement(element_type="note", measure_number=2, pitch="D4", duration=1.0),
        MusicElement(element_type="note", measure_number=3, pitch="E4", duration=1.0),
        MusicElement(element_type="note", measure_number=4, pitch="F4", duration=1.0),
    ]

    markdown = format_to_markdown(multi_measure)
    assert "Measure Count" in markdown and "4" in markdown


def test_format_with_various_durations():
    """Test formatting with notes of various durations."""
    various_durations = [
        MusicElement(element_type="note", measure_number=1, pitch="C4", duration=2.0),  # Half note
        MusicElement(element_type="note", measure_number=1, pitch="D4", duration=1.0),  # Quarter note
        MusicElement(element_type="note", measure_number=1, pitch="E4", duration=0.5),  # Eighth note
        MusicElement(element_type="note", measure_number=1, pitch="F4", duration=0.25), # Sixteenth note
    ]

    musicxml = format_to_musicxml(various_durations)
    midi = format_to_midi(various_durations)

    assert isinstance(musicxml, str)
    assert isinstance(midi, bytes)
