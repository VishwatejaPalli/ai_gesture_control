
import subprocess
import sys
import time
import os

TESTS = [
    ("Camera Test", "tests/test_camera.py"),
    ("Gesture Engine", "tests/test_camera_2.py"),
    ("Air Mouse", "tests/test_air_mouse.py"),
    ("Whiteboard", "tests/test_whiteboard.py"),
    ("Media Control", "tests/test_media.py"),
    ("Shortcut Engine", "tests/test_shortcuts.py"),
    ("Dataset Collector", "ai/collect_data.py"),
    ("Gesture Classifier", "gesture_classifier/inference.py"),
]

def run_test(name, script):
    print("\n" + "=" * 60)
    print(f"RUNNING : {name}")
    print("=" * 60)

    if not os.path.exists(script):
        print(f"✗ File not found: {script}")
        return False

    try:
        subprocess.run(
            [sys.executable, script],
            check=True
        )
        print(f"✓ {name} PASSED")
        return True

    except subprocess.CalledProcessError:
        print(f"✗ {name} FAILED")
        return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("\nGESTURE AI AUTOMATED TEST PIPELINE\n")

    passed = 0
    total = len(TESTS)

    for name, script in TESTS:
        ok = run_test(name, script)

        if ok:
            passed += 1
        else:
            print("\nStopping pipeline...")
            break

        time.sleep(2)

    print("\n" + "=" * 60)
    print("PIPELINE FINISHED")
    print(f"PASSED: {passed}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()