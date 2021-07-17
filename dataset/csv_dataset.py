import csv
from PIL import Image
import numpy
from numpy import asarray


def save_dataset():
    image = Image.open('photos/car_updated.png')
    image_sequence = asarray(image)

    writer = csv.writer(open('dataset.csv', 'w', newline=''))
    writer.writerows(image_sequence)

    data = numpy.genfromtxt('dataset.csv', dtype=int, delimiter=',')
    return data

save_dataset()
