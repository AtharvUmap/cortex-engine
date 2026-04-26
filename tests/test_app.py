from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest


# --- Tests ---

def test_app_renders_without_errors():
    """The Streamlit app should load and render without throwing any exceptions."""
    at = AppTest.from_file("app.py")
    at.run()

    # If the app raised an exception during rendering, this will fail
    assert not at.exception


def test_app_has_title():
    """The app should display a title in the main area."""
    at = AppTest.from_file("app.py")
    at.run()

    # Check that at least one title element exists
    assert len(at.title) > 0


def test_app_has_sidebar_button():
    """The sidebar should contain the ingestion trigger button."""
    at = AppTest.from_file("app.py")
    at.run()

    # Check that at least one button exists in the sidebar
    assert len(at.sidebar.button) > 0


def test_app_initializes_session_state():
    """The app should initialize chat history in session state."""
    at = AppTest.from_file("app.py")
    at.run()

    # Session state should have a chat_history key after first render
    assert "chat_history" in at.session_state


def test_app_has_chat_input():
    """The app should have a chat input field for user questions."""
    at = AppTest.from_file("app.py")
    at.run()

    # Check that a chat_input widget is present
    assert len(at.chat_input) > 0


def test_app_renders_chat_and_brain_map_tabs():
    """The main area should be split into a Chat tab and a Brain Map tab."""
    at = AppTest.from_file("app.py")
    at.run()

    # No error should be raised while rendering the tabbed layout — Streamlit
    # surfaces tab construction failures as exceptions on the AppTest object.
    assert not at.exception

    # Both tabs should be present. AppTest.tabs is the flat sequence of every
    # Tab element across the app, so we look up labels on each.
    labels = [tab.label for tab in at.tabs]
    assert "Chat" in labels
    assert "Brain Map" in labels
