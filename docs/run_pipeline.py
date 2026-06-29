import subprocess
import sys
import time

TESTS = [
    ("Camera Test", "test_camera.py"),
    ("Gesture Engine", "test_camera_2.py"),
    ("Air Mouse", "tests/test_air_mouse.py"),
    ("Whiteboard", "tests/test_whiteboard.py"),
    ("Media Control", "tests/test_media.py"),
    ("Shortcut Engine", "tests/test_shortcuts.py"),
    ("Dataset Collector", "ai/collect_data.py"),
]

def run_test(name, script):
    print("\n" + "="*60)
    print(f"RUNNING : {name}")
    print("="*60)

    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True
        )

        print(f"✓ {name} PASSED")

    except subprocess.CalledProcessError:
        print(f"✗ {name} FAILED")
        return False

    return True


def main():

    print("\nGESTURE AI AUTOMATED TEST PIPELINE\n")

    for name, script in TESTS:

        ok = run_test(name, script)

        if not ok:
            print("\nStopping pipeline...")
            break

        time.sleep(2)

    print("\nPipeline Finished")


if __name__ == "__main__":
    main()