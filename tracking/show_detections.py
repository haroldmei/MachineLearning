"""A program to show detection examples.
"""
from typing import Callable, Iterable, Tuple, Optional

import click
import cv2
import numpy as np

from detections import Detection, DETECTIONS_FACTORIES

Color = Tuple[int, int, int]

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
ALL_COLORS = [RED, GREEN, BLUE]

KEY_ESC = 27
BOX_THICKNESS = 2
IMSHOW_SLEEP_TIME = 50


def draw_detections_inplace(
    image_arr: np.ndarray,
    detections: Iterable[Detection],
    *,
    default_color: Color = GREEN,
    get_color_for_expected_id: Optional[Callable[[int], Color]] = None,
) -> np.ndarray:
    for detection in detections:
        x_min, y_min, x_max, y_max = detection.box
        if get_color_for_expected_id and detection.expected_id is not None:
            color = get_color_for_expected_id(detection.expected_id)
        else:
            color = default_color
        cv2.rectangle(
            image_arr,
            (x_min, y_min),
            (x_max, y_max),
            color=color,
            thickness=BOX_THICKNESS,
        )
    return image_arr


def show_image_and_wait(
    image_arr: np.ndarray,
    *,
    should_wait: bool = False,
    sleep_time_ms: int = IMSHOW_SLEEP_TIME,
) -> int:
    cv2.imshow("window", image_arr)

    user_input = None
    if should_wait:
        while True:
            user_input = cv2.waitKey()
            if user_input in [ord("q"), KEY_ESC]:
                break
    else:
        user_input = cv2.waitKey(sleep_time_ms)
    return user_input


@click.command(help="Show detection samples")
@click.option(
    "-t",
    "--detection-type",
    help="Type of sample to show",
    type=click.Choice(list(DETECTIONS_FACTORIES)),
    required=True,
)
def main(detection_type):
    background_image = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    current_frame = np.empty_like(background_image)
    detections_sample = DETECTIONS_FACTORIES[detection_type](background_image)

    for frame_index, detections in enumerate(detections_sample.detections):
        current_frame[...] = background_image
        draw_detections_inplace(
            current_frame,
            detections,
            get_color_for_expected_id=lambda i: ALL_COLORS[i % len(ALL_COLORS)],
        )
        user_input = show_image_and_wait(current_frame)
        if user_input == KEY_ESC:
            break


if __name__ == "__main__":
    main()
