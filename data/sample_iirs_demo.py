import numpy as np
from pathlib import Path

output = Path(
    r"C:\AstroNex\data\sample_iirs_demo.npy"
)

bands = 32
height = 256
width = 256

rng = np.random.default_rng(42)

cube = rng.random(
    (bands, height, width),
    dtype=np.float32
)

# Add several spatial patterns so the visualization
# is not completely random noise.

yy, xx = np.mgrid[
    0:height,
    0:width
]

circle1 = (
    (xx - 80) ** 2
    + (yy - 100) ** 2
) < 40 ** 2

circle2 = (
    (xx - 180) ** 2
    + (yy - 150) ** 2
) < 30 ** 2

cube[5, circle1] += 0.7
cube[15, circle2] += 0.9
cube[25, circle1] += 0.5

np.save(
    output,
    cube
)

print("Sample IIRS demo cube created:")
print(output)

print("Shape:", cube.shape)