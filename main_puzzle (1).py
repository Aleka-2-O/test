from typing import Union
import numpy as np
import cv2
import cairosvg

MASK = 'for_test\\test_mask.png'
IMAGE = 'for_test\\123.jpg'


class CropPuzzle:
    __mask: np.ndarray
    __image: np.ndarray
    __puzzle_parts: list[np.ndarray]

    def __init__(self, mask: str, image: str):
        self.__mask, self.__image = self.__image_array(mask), self.__image_array(image)
        self.__resize_and_crop_image()
        contours = self.__get_contours()
        self.__puzzle_parts = self.__cut_out_areas(contours)

    def __del__(self):
        pass

    @property
    def puzzle_parts(self) -> list[np.ndarray]:
        return self.__puzzle_parts

    @staticmethod
    def __image_array(image_path: str) -> Union[np.ndarray, None]:
        return cv2.imread(image_path)

    def __get_contours(self):
        if len(self.__mask.shape) == 3:
            gray_image = cv2.cvtColor(self.__mask, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.__mask
        _, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
        opened_image = binary_image
        contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_elements = len(contours)
        print(f'Число элементов: {num_elements} шт.')
        return contours

    def __resize_and_crop_image(self) -> None:
        target_height, target_width = self.__mask.shape[:2]
        scale = max(target_height / self.__image.shape[0], target_width / self.__image.shape[1])
        image = cv2.resize(self.__image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.__image = image[:target_height, :target_width]

    def __cut_out_areas(self, contours):
        puzzle_parts = []
        for contour in contours:
            mask = np.zeros_like(self.__image)
            cv2.drawContours(mask, [contour], -1, color=(255, 255, 255), thickness=cv2.FILLED)
            cut_out = cv2.bitwise_and(self.__image, mask)

            puzzle_parts.append(cut_out)
        return puzzle_parts


if __name__ == '__main__':
    svg_input = 'for_test\\600mask.svg'
    png_output = 'for_test\\test_mask.png'
    cairosvg.svg2png(url=svg_input, write_to=png_output)
    crop = CropPuzzle(MASK, IMAGE)
    puzzle_parts = crop.puzzle_parts
    for i, puzzle_part in enumerate(puzzle_parts, start=1):
        cv2.imwrite(f'kw\\puzzle_part_{i}.jpg', puzzle_part)
