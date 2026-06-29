"""
media_control/media_controller.py

Windows media control backend using keyboard and pyautogui.
Multimedia keys prefer the keyboard package because pyautogui
does not reliably support them on many Windows systems.

Requirements:
    pip install pyautogui keyboard
"""

import sys
import subprocess
import logging
from typing import Dict, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

IS_WINDOWS = sys.platform.startswith("win")

try:
    import pyautogui
except ImportError:
    pyautogui = None
    logging.warning(
        "pyautogui not installed. Install with: pip install pyautogui"
    )

try:
    import keyboard
except ImportError:
    keyboard = None
    logging.warning(
        "keyboard not installed. Install with: pip install keyboard"
    )


class MediaController:
    """
    Handles Windows media controls and brightness control.
    """

    def __init__(self):
        if not IS_WINDOWS:
            logging.warning(
                "MediaController is designed primarily for Windows."
            )

    # ----------------------------------------------------
    # Helper Function
    # ----------------------------------------------------
    def _press_key(self, pag_key: str, kbd_key: str) -> bool:
        """
        Presses a key using the most reliable backend.

        Multimedia keys:
            keyboard.send()

        Normal keys:
            pyautogui.press()
        """
        try:
            multimedia_keys = {
                "playpause",
                "nexttrack",
                "prevtrack",
                "volumeup",
                "volumedown",
                "volumemute",
            }

            # Multimedia keys -> keyboard module
            if pag_key in multimedia_keys and keyboard:
                logging.info(f"keyboard.send('{kbd_key}')")
                keyboard.send(kbd_key)
                return True

            # Regular keys -> pyautogui
            if pyautogui:
                logging.info(f"pyautogui.press('{pag_key}')")
                pyautogui.press(pag_key)
                return True

            # Fallback
            if keyboard:
                logging.info(f"keyboard.send('{kbd_key}')")
                keyboard.send(kbd_key)
                return True

            raise RuntimeError(
                "Neither pyautogui nor keyboard is available."
            )

        except Exception as e:
            logging.error(f"Key press failed: {e}")
            return False

    # ----------------------------------------------------
    # Media Controls
    # ----------------------------------------------------
    def play_pause(self) -> bool:
        logging.info("Executing Play/Pause")
        return self._press_key(
            "playpause",
            "play/pause media"
        )

    def next_track(self) -> bool:
        logging.info("Executing Next Track")
        return self._press_key(
            "nexttrack",
            "next track"
        )

    def previous_track(self) -> bool:
        logging.info("Executing Previous Track")
        return self._press_key(
            "prevtrack",
            "previous track"
        )

    def mute(self) -> bool:
        logging.info("Executing Mute")
        return self._press_key(
            "volumemute",
            "volume mute"
        )

    def volume_up(self) -> bool:
        logging.info("Executing Volume Up")
        return self._press_key(
            "volumeup",
            "volume up"
        )

    def volume_down(self) -> bool:
        logging.info("Executing Volume Down")
        return self._press_key(
            "volumedown",
            "volume down"
        )

    def fullscreen(self) -> bool:
        logging.info("Executing Fullscreen")
        return self._press_key(
            "f11",
            "f11"
        )

    # ----------------------------------------------------
    # Brightness Controls
    # ----------------------------------------------------
    def brightness_up(self, step=10):
        try:
            current = self._get_brightness()
            target = min(100, current + step)
            return self._set_brightness(target)
        except Exception as e:
            logging.error(e)
            return False

    def brightness_down(self, step=10):
        try:
            current = self._get_brightness()
            target = max(0, current - step)
            return self._set_brightness(target)
        except Exception as e:
            logging.error(e)
            return False

    # ----------------------------------------------------
    # Command Router
    # ----------------------------------------------------
    def execute(self, command: str) -> bool:
        command = command.strip().lower().replace(" ", "_")

        commands: Dict[str, Callable[[], bool]] = {
            "play_pause": self.play_pause,
            "play/pause": self.play_pause,

            "next_track": self.next_track,
            "next": self.next_track,

            "previous_track": self.previous_track,
            "prev_track": self.previous_track,
            "previous": self.previous_track,

            "mute": self.mute,

            "volume_up": self.volume_up,
            "volume_down": self.volume_down,

            "fullscreen": self.fullscreen,

            "brightness_up":
                lambda: self.brightness_up(),

            "brightness_down":
                lambda: self.brightness_down(),
        }

        if command not in commands:
            logging.warning(
                f"Unsupported command: {command}"
            )
            return False

        logging.info(
            f"Routing command: {command}"
        )

        return commands[command]()

    # ----------------------------------------------------
    # Brightness
    # ----------------------------------------------------
    def _get_brightness(self) -> int:
        if not IS_WINDOWS:
            return 50

        queries = [
            "(Get-CimInstance -Namespace root/WMI "
            "-ClassName WmiMonitorBrightness)"
            ".CurrentBrightness",

            "(Get-WmiObject -Namespace root/WMI "
            "-Class WmiMonitorBrightness)"
            ".CurrentBrightness",
        ]

        for q in queries:
            try:
                result = subprocess.run(
                    f'powershell -Command "{q}"',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=3
                )

                if result.returncode == 0:
                    out = result.stdout.strip()
                    if out.isdigit():
                        return int(out)

            except Exception:
                pass

        return 50

    def _set_brightness(self, level: int) -> bool:
        if not IS_WINDOWS:
            return False

        level = max(0, min(100, level))

        queries = [
            f'''
            Invoke-CimMethod
            -InputObject
            (Get-CimInstance
            -Namespace root/WMI
            -ClassName
            WmiMonitorBrightnessMethods)
            -MethodName WmiSetBrightness
            -Arguments
            @{{Brightness={level};Timeout=0}}
            ''',

            f'''
            (Get-WmiObject
            -Namespace root/WMI
            -Class
            WmiMonitorBrightnessMethods)
            .WmiSetBrightness(1,{level})
            '''
        ]

        for q in queries:
            try:
                result = subprocess.run(
                    f'powershell -Command "{q}"',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=4
                )

                if result.returncode == 0:
                    logging.info(
                        f"Brightness set to {level}%"
                    )
                    return True

            except Exception:
                pass

        logging.warning(
            "Brightness control unsupported."
        )
        return False


if __name__ == "__main__":
    import time

    mc = MediaController()

    print("Testing play/pause in 3 seconds...")
    time.sleep(3)
    mc.play_pause()

    print("Testing next track...")
    time.sleep(2)
    mc.next_track()

    print("Testing previous track...")
    time.sleep(2)
    mc.previous_track()

    print("Testing volume up...")
    time.sleep(2)
    mc.volume_up()

    print("Testing volume down...")
    time.sleep(2)
    mc.volume_down()

    print("Done.")