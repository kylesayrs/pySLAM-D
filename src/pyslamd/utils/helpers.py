from typing import List, Any

import os
import parse
import numpy


def get_image_paths(image_dir: str) -> List[str]:
    file_names = [
        file_name
        for file_name in os.listdir(image_dir)
        if os.path.splitext(file_name)[1].lower() == ".jpg"
    ]

    file_numbers = [
        int(parse.parse("img_{}.jpg", file_name.lower())[0])
        for file_name in file_names
    ]

    file_names_sorted = list(zip(
        *sorted(
            zip(file_names, file_numbers),
            key=lambda pair: pair[1]
        )
    ))[0]

    file_paths = [
        os.path.join(image_dir, file_name)
        for file_name in file_names_sorted
    ]

    return file_paths


def all_none(list_: List[Any]):
    return all(value is None for value in list_)


def mask_list(list_: List[Any], mask: numpy.ndarray) -> List[Any]:
    return [
        element
        for index, element in enumerate(list_)
        if mask[index]
    ]