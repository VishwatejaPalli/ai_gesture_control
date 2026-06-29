# gesture_classifier/classifier.py


class GestureClassifier:
    """
    Rule-based gesture classifier.
    Input:
        fingers = {
            "thumb": bool,
            "index": bool,
            "middle": bool,
            "ring": bool,
            "pinky": bool
        }

        pinch = bool

    Output:
        gesture name string
    """

    def __init__(self):
        pass

    def classify(self, fingers, pinch=False):

        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]

        # Pinch gesture
        if pinch:
            return "pinch"

        # Fist
        if not any([
            thumb,
            index,
            middle,
            ring,
            pinky
        ]):
            return "fist"

        # Open palm
        if all([
            thumb,
            index,
            middle,
            ring,
            pinky
        ]):
            return "palm"

        # Pointing
        if (
            index and
            not middle and
            not ring and
            not pinky
        ):
            return "pointing"

        # Peace sign
        if (
            index and
            middle and
            not ring and
            not pinky
        ):
            return "peace"

        # Thumbs up
        if (
            thumb and
            not index and
            not middle and
            not ring and
            not pinky
        ):
            return "thumbs_up"

        # Three fingers
        if (
            index and
            middle and
            ring and
            not pinky
        ):
            return "three"

        # Four fingers
        if (
            index and
            middle and
            ring and
            pinky and
            not thumb
        ):
            return "four"

        return "unknown"


if __name__ == "__main__":

    classifier = GestureClassifier()

    test = {
        "thumb": False,
        "index": True,
        "middle": False,
        "ring": False,
        "pinky": False
    }

    print(classifier.classify(test))