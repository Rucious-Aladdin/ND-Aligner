def convert_pad_shape(pad_shape):  # pyright: ignore
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape
