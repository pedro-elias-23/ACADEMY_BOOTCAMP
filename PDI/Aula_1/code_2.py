import numpy as np
from pdi_utils import load_flipped_seville, show_image
import imutils


flipped_seville = load_flipped_seville()

# Show original image
show_image(flipped_seville, 'Seville Flipped')

# Flip the image vertically
seville_vertical_flip = np.flipud(flipped_seville)

# Show image flippped vertically
show_image(seville_vertical_flip, 'Seville Vertical Flipped')

# Flip the image horizontally

seville_horizontal_flip = np.fliplr(seville_vertical_flip)

# Show image flipped horizontally
show_image(seville_horizontal_flip, 'Seville horizontal Flipped')
