"""
Unit tests for grounding.omr_parser module.

Tests OMR parsing functionality using mocked subprocess calls to avoid requiring
Audiveris installation for test execution.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from grounding.omr_parser import (
    MusicElement,
    AudiverisOMRError,
    parse_music_pdf,
    detect_music_content,
    _find_audiveris_binary,
    _check_java_version,
)


# Fixtures


@pytest.fixture
def temp_pdf(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_file = tmp_path / "test_music.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%...")  # Minimal PDF header
    return pdf_file


@pytest.fixture
def temp_musicxml(tmp_path):
    """Create a temporary MusicXML file for testing."""
    musicxml_file = tmp_path / "test_music.mxl"
    # Create a minimal MusicXML structure
    musicxml_content = b"PK\x03\x04"  # Minimal .mxl (compressed MusicXML) header
    musicxml_file.write_bytes(musicxml_content)
    return musicxml_file


# MusicElement Tests


def test_music_element_creation_note():
    """Test creating a MusicElement for a note."""
    elem = MusicElement(
        element_type="note",
        measure_number=1,
        pitch="C4",
        duration=1.0
    )

    assert elem.element_type == "note"
    assert elem.measure_number == 1
    assert elem.pitch == "C4"
    assert elem.duration == 1.0
    assert elem.staff_number == 1  # Default
    assert elem.voice_number == 1  # Default
    assert elem.metadata == {}


def test_music_element_creation_rest():
    """Test creating a MusicElement for a rest."""
    elem = MusicElement(
        element_type="rest",
        measure_number=2,
        duration=0.5
    )

    assert elem.element_type == "rest"
    assert elem.measure_number == 2
    assert elem.duration == 0.5
    assert elem.pitch is None


def test_music_element_str_note():
    """Test MusicElement string representation for a note."""
    elem = MusicElement(
        element_type="note",
        measure_number=1,
        pitch="C4",
        duration=1.0
    )

    assert str(elem) == "Note(C4, 1.0q, m1)"


def test_music_element_str_rest():
    """Test MusicElement string representation for a rest."""
    elem = MusicElement(
        element_type="rest",
        measure_number=2,
        duration=0.5
    )

    assert str(elem) == "Rest(0.5q, m2)"


def test_music_element_str_clef():
    """Test MusicElement string representation for a clef."""
    elem = MusicElement(
        element_type="clef",
        measure_number=1
    )

    assert str(elem) == "Clef(m1)"


# Java Version Check Tests


@patch('grounding.omr_parser.subprocess.run')
def test_check_java_version_success(mock_run):
    """Test Java version check when Java is installed."""
    mock_run.return_value = Mock(
        returncode=0,
        stderr='openjdk version "24.0.2" 2025-03-18',
        stdout=''
    )

    assert _check_java_version() is True


@patch('grounding.omr_parser.subprocess.run')
def test_check_java_version_not_found(mock_run):
    """Test Java version check when Java is not installed."""
    mock_run.side_effect = FileNotFoundError("java not found")

    assert _check_java_version() is False


# Audiveris Binary Detection Tests


@patch('grounding.omr_parser.os.getenv')
@patch('grounding.omr_parser.Path.exists')
def test_find_audiveris_binary_env_var(mock_exists, mock_getenv):
    """Test finding Audiveris via AUDIVERIS_HOME environment variable."""
    mock_getenv.return_value = "/usr/local/audiveris"
    mock_exists.return_value = True

    binary = _find_audiveris_binary()

    assert binary == Path("/usr/local/audiveris/Audiveris")


@patch('grounding.omr_parser.os.getenv')
@patch('grounding.omr_parser.Path.exists')
def test_find_audiveris_binary_macos_default(mock_exists, mock_getenv):
    """Test finding Audiveris at macOS default location."""
    mock_getenv.return_value = None  # No env var

    # Mock exists to return True for macOS path, False otherwise
    def exists_side_effect():
        # This is called on the Path object instance
        return True

    # Set return values: False for env var path, True for macOS path
    mock_exists.side_effect = [True]  # macOS path exists

    binary = _find_audiveris_binary()

    assert binary == Path("/Applications/Audiveris.app/Contents/MacOS/Audiveris")


@patch('grounding.omr_parser.os.getenv')
@patch('grounding.omr_parser.Path.exists')
@patch('grounding.omr_parser.subprocess.run')
def test_find_audiveris_binary_not_found(mock_run, mock_exists, mock_getenv):
    """Test when Audiveris is not found anywhere."""
    mock_getenv.return_value = None
    mock_exists.return_value = False
    mock_run.return_value = Mock(returncode=1, stdout='')

    binary = _find_audiveris_binary()

    assert binary is None


# parse_music_pdf Tests


@patch('grounding.omr_parser._check_java_version')
@patch('grounding.omr_parser._find_audiveris_binary')
@patch('grounding.omr_parser.subprocess.run')
@patch('grounding.omr_parser._parse_musicxml_to_elements')
def test_parse_music_pdf_success(mock_parse_xml, mock_subprocess, mock_find_audiveris, mock_check_java, temp_pdf, tmp_path):
    """Test successful parsing of a music PDF."""
    # Setup mocks
    mock_check_java.return_value = True
    mock_find_audiveris.return_value = Path("/usr/bin/audiveris")
    mock_subprocess.return_value = Mock(returncode=0, stderr='', stdout='Success')

    # Create fake MusicXML output
    musicxml_file = tmp_path / "test_music.mxl"
    musicxml_file.write_bytes(b"fake musicxml")

    # Mock music21 parsing to return sample elements
    mock_parse_xml.return_value = [
        MusicElement(element_type="note", measure_number=1, pitch="C4", duration=1.0)
    ]

    # Mock glob to find the musicxml file
    with patch.object(Path, 'glob', return_value=[musicxml_file]):
        elements = parse_music_pdf(temp_pdf, output_dir=tmp_path)

    assert len(elements) == 1
    assert elements[0].pitch == "C4"


@patch('grounding.omr_parser._check_java_version')
def test_parse_music_pdf_java_not_found(mock_check_java, temp_pdf):
    """Test parse_music_pdf raises error when Java is not installed."""
    mock_check_java.return_value = False

    with pytest.raises(AudiverisOMRError, match="Java Runtime Environment"):
        parse_music_pdf(temp_pdf)


@patch('grounding.omr_parser._check_java_version')
@patch('grounding.omr_parser._find_audiveris_binary')
def test_parse_music_pdf_audiveris_not_found(mock_find_audiveris, mock_check_java, temp_pdf):
    """Test parse_music_pdf raises error when Audiveris is not installed."""
    mock_check_java.return_value = True
    mock_find_audiveris.return_value = None

    with pytest.raises(AudiverisOMRError, match="Audiveris not found"):
        parse_music_pdf(temp_pdf)


def test_parse_music_pdf_file_not_found():
    """Test parse_music_pdf raises error when PDF file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        parse_music_pdf(Path("/nonexistent/file.pdf"))


def test_parse_music_pdf_not_a_pdf(tmp_path):
    """Test parse_music_pdf raises error when file is not a PDF."""
    text_file = tmp_path / "test.txt"
    text_file.write_text("Not a PDF")

    with pytest.raises(ValueError, match="not a PDF"):
        parse_music_pdf(text_file)


@patch('grounding.omr_parser._check_java_version')
@patch('grounding.omr_parser._find_audiveris_binary')
@patch('grounding.omr_parser.subprocess.run')
def test_parse_music_pdf_audiveris_fails(mock_subprocess, mock_find_audiveris, mock_check_java, temp_pdf):
    """Test parse_music_pdf handles Audiveris processing failure."""
    mock_check_java.return_value = True
    mock_find_audiveris.return_value = Path("/usr/bin/audiveris")
    mock_subprocess.return_value = Mock(returncode=1, stderr='Processing failed', stdout='')

    with pytest.raises(AudiverisOMRError, match="processing failed"):
        parse_music_pdf(temp_pdf)


@patch('grounding.omr_parser._check_java_version')
@patch('grounding.omr_parser._find_audiveris_binary')
@patch('grounding.omr_parser.subprocess.run')
def test_parse_music_pdf_timeout(mock_subprocess, mock_find_audiveris, mock_check_java, temp_pdf):
    """Test parse_music_pdf handles timeout."""
    import subprocess

    mock_check_java.return_value = True
    mock_find_audiveris.return_value = Path("/usr/bin/audiveris")
    mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=120)

    with pytest.raises(AudiverisOMRError, match="timed out"):
        parse_music_pdf(temp_pdf)


@patch('grounding.omr_parser._check_java_version')
@patch('grounding.omr_parser._find_audiveris_binary')
@patch('grounding.omr_parser.subprocess.run')
def test_parse_music_pdf_no_musicxml_output(mock_subprocess, mock_find_audiveris, mock_check_java, temp_pdf, tmp_path):
    """Test parse_music_pdf handles case where no MusicXML is generated."""
    mock_check_java.return_value = True
    mock_find_audiveris.return_value = Path("/usr/bin/audiveris")
    mock_subprocess.return_value = Mock(returncode=0, stderr='', stdout='Success')

    # No MusicXML files in output dir (glob returns empty list)
    with patch.object(Path, 'glob', return_value=[]):
        with pytest.raises(AudiverisOMRError, match="No MusicXML output generated"):
            parse_music_pdf(temp_pdf, output_dir=tmp_path)


# detect_music_content Tests


@patch('grounding.omr_parser._detect_staff_lines_quick')
def test_detect_music_content_quick_true(mock_detect_staff, temp_pdf):
    """Test detect_music_content with quick heuristic returning True."""
    mock_detect_staff.return_value = True

    result = detect_music_content(temp_pdf, quick=True)

    assert result is True


@patch('grounding.omr_parser._detect_staff_lines_quick')
def test_detect_music_content_quick_false(mock_detect_staff, temp_pdf):
    """Test detect_music_content with quick heuristic returning False."""
    mock_detect_staff.return_value = False

    result = detect_music_content(temp_pdf, quick=True)

    assert result is False


@patch('grounding.omr_parser._detect_with_audiveris_sample')
def test_detect_music_content_audiveris_sample(mock_detect_audiveris, temp_pdf):
    """Test detect_music_content with Audiveris sampling."""
    mock_detect_audiveris.return_value = True

    result = detect_music_content(temp_pdf, quick=False)

    assert result is True


def test_detect_music_content_file_not_found():
    """Test detect_music_content returns False for non-existent file."""
    result = detect_music_content(Path("/nonexistent/file.pdf"))

    assert result is False


@patch('grounding.omr_parser.parse_music_pdf')
def test_detect_with_audiveris_sample_success(mock_parse):
    """Test _detect_with_audiveris_sample when parsing succeeds."""
    from grounding.omr_parser import _detect_with_audiveris_sample

    mock_parse.return_value = [
        MusicElement(element_type="note", measure_number=1, pitch="C4", duration=1.0)
    ]

    result = _detect_with_audiveris_sample(Path("fake.pdf"))

    assert result is True


@patch('grounding.omr_parser.parse_music_pdf')
def test_detect_with_audiveris_sample_failure(mock_parse):
    """Test _detect_with_audiveris_sample when parsing fails."""
    from grounding.omr_parser import _detect_with_audiveris_sample

    mock_parse.side_effect = AudiverisOMRError("Not music")

    result = _detect_with_audiveris_sample(Path("fake.pdf"))

    assert result is False


# Integration-like Test (still mocked but tests full flow)


@patch('grounding.omr_parser._check_java_version')
@patch('grounding.omr_parser._find_audiveris_binary')
@patch('grounding.omr_parser.subprocess.run')
@patch('music21.converter.parse')  # Mock music21 converter.parse
def test_parse_music_pdf_full_flow(mock_converter_parse, mock_subprocess, mock_find_audiveris, mock_check_java, temp_pdf, tmp_path):
    """Test full parsing flow with mocked music21."""
    # Setup mocks
    mock_check_java.return_value = True
    mock_find_audiveris.return_value = Path("/usr/bin/audiveris")
    mock_subprocess.return_value = Mock(returncode=0, stderr='', stdout='Success')

    # Create fake MusicXML output
    musicxml_file = tmp_path / "test_music.mxl"
    musicxml_file.write_bytes(b"fake musicxml")

    # Mock music21 score
    mock_score = MagicMock()
    mock_part = MagicMock()
    mock_measure = MagicMock()
    mock_note = MagicMock()
    mock_note.isNote = True
    mock_note.isRest = False
    mock_note.nameWithOctave = "C4"
    mock_note.quarterLength = 1.0

    mock_measure.notesAndRests = [mock_note]
    mock_measure.getElementsByClass.return_value = [mock_measure]
    mock_part.getElementsByClass.return_value = [mock_measure]
    mock_score.parts = [mock_part]
    mock_converter_parse.return_value = mock_score

    # Mock glob to find the musicxml file
    with patch.object(Path, 'glob', return_value=[musicxml_file]):
        elements = parse_music_pdf(temp_pdf, output_dir=tmp_path)

    assert len(elements) == 1
    assert elements[0].element_type == "note"
    assert elements[0].pitch == "C4"
    assert elements[0].duration == 1.0
