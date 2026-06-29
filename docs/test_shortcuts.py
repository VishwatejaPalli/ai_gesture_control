"""
tests/test_shortcuts.py

Unit test suite verifying state, registration, storage actions, and 
execution cooldowns of the MacroEngine.
"""

import os
import sys
import unittest
import tempfile
import time

# Ensure project root is in sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gesture_shortcuts.macro_engine import MacroEngine


class TestMacroEngine(unittest.TestCase):
    """Verifies registration modifications and mapping actions."""

    def setUp(self):
        # Configure with a small cooldown boundary for execution testing
        self.engine = MacroEngine(default_cooldown=0.1)

    def test_default_config_mappings(self):
        """Ensures default gestural associations are registered on launch."""
        self.assertEqual(self.engine.mappings.get("thumbs_up"), "open_browser")
        self.assertEqual(self.engine.mappings.get("peace"), "copy")
        self.assertEqual(self.engine.mappings.get("three"), "paste")
        self.assertEqual(self.engine.mappings.get("four"), "save")
        self.assertEqual(self.engine.mappings.get("fist"), "undo")
        self.assertEqual(self.engine.mappings.get("palm"), "stop")

    def test_register_and_remove_methods(self):
        """Verifies runtime dictionary modifications."""
        # Add new macro association
        self.engine.register_macro("ok", "confirm")
        self.assertEqual(self.engine.mappings.get("ok"), "confirm")

        # Delete gesture macro association
        status = self.engine.remove_macro("ok")
        self.assertTrue(status)
        self.assertNotIn("ok", self.engine.mappings)

        # Attempting to delete non-existent key returns False
        status_fail = self.engine.remove_macro("non_existent_key")
        self.assertFalse(status_fail)

    def test_execution_cooldown_timers(self):
        """Tests that rapid triggers are correctly blocked by cooldown filters."""
        # First execution must trigger the mapped action string
        action_1 = self.engine.execute("thumbs_up")
        self.assertEqual(action_1, "open_browser")

        # Second immediate trigger should be blocked (returns None)
        action_2 = self.engine.execute("thumbs_up")
        self.assertIsNone(action_2)
        self.assertEqual(self.engine.history[-1]["status"], "ignored_cooldown")

        # Delay execution until cooldown expires
        time.sleep(0.12)
        action_3 = self.engine.execute("thumbs_up")
        self.assertEqual(action_3, "open_browser")
        self.assertEqual(self.engine.history[-1]["status"], "executed")

    def test_serialization_persistence(self):
        """Verifies loading and saving JSON configuration files."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Register a custom key and save configuration
            self.engine.register_macro("custom_test_gesture", "run_test_assertion")
            self.engine.save_config(temp_path)

            # Load into a clean engine instance
            new_engine = MacroEngine()
            new_engine.load_config(temp_path)

            self.assertEqual(new_engine.mappings.get("custom_test_gesture"), "run_test_assertion")
            self.assertEqual(new_engine.default_cooldown, 0.1)

        finally:
            # Remove temp files safely
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()