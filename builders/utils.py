import hashlib
from PIL import Image, ImageDraw

def calculate_md5(file_path, chunk_size=8192):
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        chunk = f.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = f.read(chunk_size)

    return md5.hexdigest()

class BoundingBox:
    def __init__(self, x1, y1, x2, y2, d=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.d = d

    @staticmethod
    def from_dict(d):
        return BoundingBox(
            d["@xmin"],
            d["@ymin"],
            d["@xmax"],
            d["@ymax"],
            d,
        )

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def tuple(self):
        return (
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        )

    def __add__(self, a):
        x1 = min(self.x1, a.x1)
        y1 = min(self.y1, a.y1)
        x2 = max(self.x2, a.x2)
        y2 = max(self.y2, a.y2)
        return BoundingBox(x1, y1, x2, y2)

    def __mul__(self, a):
        x1_int = max(self.x1, a.x1)
        y1_int = max(self.y1, a.y1)
        x2_int = min(self.x2, a.x2)
        y2_int = min(self.y2, a.y2)

        if x1_int < x2_int and y1_int < y2_int:
            return BoundingBox(x1_int, y1_int, x2_int, y2_int)
        else:
            return BoundingBox(0,0,0,0)

    def __repr__(self):
        return f"BoundingBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"

def arrange_images_right_to_left_without_blank(top_images, bottom_images, margin=40, is_rightfirst=True):
    if is_rightfirst:
        top_images = list(reversed(top_images))
        bottom_images = list(reversed(bottom_images))

    # Ensure there are exactly 3 images in each row
    assert len(top_images) == 3, "Top row should have exactly 3 images"
    assert len(bottom_images) == 3, "Bottom row should have exactly 3 images"

    # Calculate the width and height of the canvas
    canvas_top_width = sum([img.width for img in top_images]) + (margin * 4)
    canvas_bottom_width = sum([img.width for img in bottom_images]) + (margin * 4)
    canvas_width = max(canvas_top_width, canvas_bottom_width)

    max_top_height = max([img.height for img in top_images])
    max_bottom_height = max([img.height for img in bottom_images])
    canvas_height = max_top_height + max_bottom_height + (margin * 3)

    # Create the canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

    # Paste top row images (right to left)
    x_offset = canvas_width - margin
    for img in reversed(top_images):
        x_offset -= img.width
        canvas.paste(img, (x_offset, margin))
        x_offset -= margin

    # Paste bottom row images (right to left)
    x_offset = canvas_width - margin
    y_offset = max_top_height + (2 * margin)
    for img in reversed(bottom_images):
        x_offset -= img.width
        canvas.paste(img, (x_offset, y_offset))
        x_offset -= margin

    return canvas

def arrange_images_right_to_left_with_borders(top_images, bottom_images, margin=40, border_thickness=30):
    top_images = list(reversed(top_images))
    bottom_images = list(reversed(bottom_images))

    # Ensure there are exactly 3 images in each row
    assert len(top_images) == 3, "Top row should have exactly 3 images"
    assert len(bottom_images) == 3, "Bottom row should have exactly 3 images"

    # Calculate the width and height of the canvas
    canvas_top_width = sum([img.width for img in top_images]) + (margin * 4)
    canvas_bottom_width = sum([img.width + 2 * border_thickness for img in bottom_images]) + (margin * 4)
    canvas_width = max(canvas_top_width, canvas_bottom_width)

    max_top_height = max([img.height for img in top_images])
    max_bottom_height = max([img.height + 2 * border_thickness for img in bottom_images])
    canvas_height = max_top_height + max_bottom_height + (margin * 3)

    # Create the canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

    # Paste top row images (right to left)
    x_offset = canvas_width - margin
    for img in reversed(top_images):
        x_offset -= img.width
        canvas.paste(img, (x_offset, margin))
        x_offset -= margin

    # Paste bottom row images with borders (right to left)
    border_colors = ['red', 'green', 'blue']  # Border colors for each bottom image
    x_offset = canvas_width - margin
    y_offset = max_top_height + (2 * margin)
    for img, color in zip(reversed(bottom_images), border_colors):
        # Create a new image with border
        bordered_img = Image.new('RGB', (img.width + 2 * border_thickness, img.height + 2 * border_thickness), color)
        bordered_img.paste(img, (border_thickness, border_thickness))

        x_offset -= img.width + 2 * border_thickness
        canvas.paste(bordered_img, (x_offset, y_offset))
        x_offset -= margin

    return canvas

def cyclic_permutation(arr, n):
    n = n % len(arr)
    return list(arr[-n:] + arr[:-n])

def make_choice_string(i_displacement, l_choices, i_correct_base=0):
    choices = cyclic_permutation(l_choices, i_displacement)
    indices = list("ABCDEFGHIJKLMNOP")
    l_choices = []
    for index, choice in zip(indices, choices):
        l_choices.append(f"{index}. {choice}")

    s_choice = "\n".join(l_choices)

    i_correct_shifted = (i_correct_base + i_displacement) % len(l_choices)
    expected_alphabet = indices[i_correct_shifted]

    return s_choice, choices, expected_alphabet

def chunk_generator(input_iter, chunk_size=5):
    input_list = list(input_iter)
    for i in range(0, len(input_list), chunk_size):
        ret = input_list[i:i + chunk_size]
        if len(ret) == chunk_size:
            yield ret
        else:
            break

def crop_polygon(image, polygon):
    draw = ImageDraw.Draw(image)
    draw.polygon(polygon, outline="black", fill="black")
    return image

def debug():
    import ipdb;
    ipdb.set_trace()
