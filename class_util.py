import numpy

class_names = [
    'clutter',
    'building',
    'grass',
    'pond',
    'road',
    'dirt',
    'tree',
    'vehicle',
    'sign',
    'car',
    'guardrail',
]
class_colors = numpy.array([
    [50,50,50],
    [128,0,0],
    [0,128,0],
    [0,0,128],
    [128,64,128],
    [60,40,222],
    [128,128,0],
    [64,0,128],
    [192,128,128],
    [255, 255, 0],
    [255, 165, 0],
], dtype=numpy.uint8)

if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt
    import skimage.io

    parser = argparse.ArgumentParser("")
    parser.add_argument( '--image', type=str, help='')
    parser.add_argument( '--label', type=str, help='')
    FLAGS = parser.parse_args()

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


    img = skimage.io.imread(FLAGS.image)
    label = skimage.io.imread(FLAGS.label)
    label_viz = img//2 + label//2
    plt.imshow(label_viz)
    plt.axis('off')

    label_int = numpy.zeros((label.shape[0], label.shape[1]), dtype=int)
    for i in range(len(class_colors)):
        matched = numpy.all(label == class_colors[i], axis=-1)
        label_int[matched] = i
    label_set = set(label_int.flatten())
    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(class_names):
        if label_value not in label_set:
            continue
        fc = class_colors[label_value] / 255.0
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append('{value}: {name}'
                          .format(value=label_value, name=label_name))
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)
    plt.show()
