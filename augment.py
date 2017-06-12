import numpy as np
import cv2

def jitter_image(image, scale_factor=1, translation_limit=30, rotation_limit=180):
  '''
  options
  --------
  flips
  crops
  color noise
  brightness noise
  rotation
  translation
  zoom
  blur
  '''

  rows, columns = image.shape

  angle = np.random.randint(-rotation_limit, rotation_limit+1)
  rotation_matrix = cv2.getRotationMatrix2D((columns/2, rows/2), angle, 1)

  x_translate = np.random.randint(-translation_limit, translation_limit+1)
  y_translate = np.random.randint(-translation_limit, translation_limit+1)
  translation_matrix = np.float32([[1, 0, x_translate],
                                   [0, 1, y_translate]])

  rotated = cv2.warpAffine(image, rotation_matrix, (columns, rows))
  translated = cv2.warpAffine(rotated, translation_matrix, (columns, rows))
  # resized = cv2.resize(translated, (columns*scale_factor, rows*scale_factor))

  return translated


def augmented_batch_generator(data, labels=None, batch_size=25):
  starting_point = 0

  while starting_point < len(data):
    batch = data[starting_point:starting_point+batch_size]
    yield [jitter_image(image) for image in batch]
    starting_point += batch_size

if __name__ == '__main__':
  data = np.rollaxis(np.load('data/inputs_valence.npy'), 2)
  data /= np.expand_dims(np.expand_dims(data.max(axis=(1, 2)), 1), 2)

  for image in data:
    cv2.imshow('test', np.hstack([image, jitter_image(image)]))
    cv2.waitKey(0)
