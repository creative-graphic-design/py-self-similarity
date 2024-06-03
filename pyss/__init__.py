from typing import Optional

import numpy as np
from scipy import ndimage
from skimage import color


def computeWeightedDistances(
    descriptor, descriptorglobal, bins, levels, method, SESFW, section, descriptornn
):
    distances = []

    if SESFW == "G":
        comparisonglobal = descriptor[:bins]

        temp = np.zeros((levels, 2), dtype=int)
        temp[0, 0] = bins + 1
        temp[0, 1] = section**2 * bins + temp[0, 0] - 1

        for i in range(1, levels):
            temp[i, 0] = temp[i - 1, 1] + 1
            temp[i, 1] = section ** (2 * (i + 1)) * bins + temp[i, 0] - 1

        for i in range(levels):
            for j in range(temp[i, 0] - 1, temp[i, 1], bins):
                part = descriptor[j : j + bins]

                if np.max(comparisonglobal) > 1e-8 and np.max(part) > 1e-8:
                    dist1 = np.sum(np.minimum(comparisonglobal, part))

                    m1 = np.mean(descriptornn[:bins])
                    m2 = np.mean(descriptornn[j : j + bins])

                    area = section ** (2 * i)
                    m2 = m2 * area

                    if m1 < 1e-8 or m2 < 1e-8:
                        strengthsimilarity = 0
                    elif m1 > m2:
                        strengthsimilarity = m2 / m1
                    else:
                        strengthsimilarity = m1 / m2

                    dist1 = dist1 * strengthsimilarity
                    distances.append(dist1)
                else:
                    distances.append(0)

    # The other cases (SESFW == 'N' and SESFW == 'P') are not implemented
    # because the Matlab code is commented out

    return distances


def computeSD(descriptor, bins, levels, section):
    """
    Computes standard deviation of the descriptor

    Parameters:
    descriptor -- the descriptor
    bins -- number of bins
    levels -- number of levels
    section -- section

    Returns:
    standard deviation values
    """

    temp = np.zeros((levels, 2), dtype=int)
    temp[0, 0] = bins
    temp[0, 1] = section**2 * bins + temp[0, 0] - 1

    for i in range(1, levels):
        temp[i, 0] = temp[i - 1, 1] + 1
        temp[i, 1] = section ** (2 * (i + 1)) * bins + temp[i, 0] - 1

    descript = descriptor[temp[levels - 1, 0] - 1 : temp[levels - 1, 1]]
    sd = np.std(descript)

    if np.max(descript) < 1e-8:
        sd = np.nan

    sdvalues = sd

    return sdvalues


def normalizeDescriptorGlobal(descriptor):
    """
    Globally normalizes the descriptor

    Parameters:
    descriptor -- the descriptor to be globally normalized

    Returns:
    globally normalized descriptor
    """

    if np.sum(descriptor) != 0:
        normalized_descriptor_global = descriptor / np.sum(descriptor)

    return normalized_descriptor_global


def normalizeDescriptor(descriptor, bins):
    """
    Normalizes the descriptor

    Parameters:
    descriptor -- the descriptor to be normalized
    bins -- number of bins

    Returns:
    normalized descriptor
    """

    b = np.reshape(descriptor, (bins, len(descriptor) // bins))
    c = np.sum(b, axis=0)

    temp = np.zeros(b.shape)

    for i in range(b.shape[1]):
        if c[i] != 0:
            temp[:, i] = b[:, i] / c[i]
        else:
            temp[:, i] = b[:, i]

    normalized_descriptor = np.reshape(temp, len(descriptor))

    return normalized_descriptor


def computeDescriptorGlobal(GradientValue, GradientAngle, bins, angle, levels, section):
    """
    Computes Pyramid Histogram of Oriented Gradient over 'levels' pyramid
    levels using gradient values and directions

    Parameters:
    GradientValue -- matrix of gradient values
    GradientAngle -- matrix of gradient directions
    bins -- Number of bins on the histogram, needs to be a multiple of 3
    angle -- 180 or 360
    levels -- number of pyramid levels

    Returns:
    pyramid histogram of oriented gradients (phog descriptor)
    """

    descriptor = []

    intervalSize = angle / bins
    halfIntervalSize = intervalSize / 2

    # level 0
    ind = (GradientAngle >= angle - halfIntervalSize) | (GradientAngle < halfIntervalSize)
    descriptor.append(np.sum(GradientValue[ind]))

    for b in range(1, bins):
        ind = (GradientAngle >= b * intervalSize - halfIntervalSize) & (
            GradientAngle < (b + 1) * intervalSize - halfIntervalSize
        )
        descriptor.append(np.sum(GradientValue[ind]))

    # local normalization
    descriptor = normalizeDescriptor(descriptor, bins)

    # other levels
    for l in range(1, levels + 1):
        cellSizeX = GradientAngle.shape[1] // (section**l)
        cellSizeY = GradientAngle.shape[0] // (section**l)

        if cellSizeX < 1 or cellSizeY < 1:
            raise ValueError("Cell size < 1, adjust number of levels")

        for j in range(section**l):
            leftX = int(j * cellSizeX)
            rightX = int((j + 1) * cellSizeX)

            for i in range(section**l):
                topY = int(i * cellSizeY)
                bottomY = int((i + 1) * cellSizeY)

                GradientValueCell = GradientValue[topY:bottomY, leftX:rightX]
                GradientAngleCell = GradientAngle[topY:bottomY, leftX:rightX]

                ind = (GradientAngleCell >= angle - halfIntervalSize) | (
                    GradientAngleCell < halfIntervalSize
                )
                local_descriptor = [np.sum(GradientValueCell[ind])]

                for b in range(1, bins):
                    ind = (GradientAngleCell >= b * intervalSize - halfIntervalSize) & (
                        GradientAngleCell < (b + 1) * intervalSize - halfIntervalSize
                    )
                    local_descriptor.append(np.sum(GradientValueCell[ind]))

                local_descriptor = normalizeDescriptor(local_descriptor, bins)
                descriptor.extend(local_descriptor)

    descriptor = normalizeDescriptorGlobal(descriptor)

    return descriptor


def computeDescriptor(GradientValue, GradientAngle, bins, angle, levels, section):
    """
    Computes Pyramid Histogram of Oriented Gradient over 'levels' pyramid
    levels using gradient values and directions

    Parameters:
    GradientValue -- matrix of gradient values
    GradientAngle -- matrix of gradient directions
    bins -- Number of bins on the histogram, needs to be a multiple of 3
    angle -- 180 or 360
    levels -- number of pyramid levels

    Returns:
    pyramid histogram of oriented gradients (phog descriptor)
    """

    descriptor = []

    intervalSize = angle / bins
    halfIntervalSize = intervalSize / 2

    # level 0
    ind = (GradientAngle >= angle - halfIntervalSize) | (GradientAngle < halfIntervalSize)
    descriptor.append(np.sum(GradientValue[ind]))

    for b in range(1, bins):
        ind = (GradientAngle >= b * intervalSize - halfIntervalSize) & (
            GradientAngle < (b + 1) * intervalSize - halfIntervalSize
        )
        descriptor.append(np.sum(GradientValue[ind]))

    # other levels
    for l in range(1, levels + 1):
        cellSizeX = GradientAngle.shape[1] // (section**l)
        cellSizeY = GradientAngle.shape[0] // (section**l)

        if cellSizeX < 1 or cellSizeY < 1:
            raise ValueError("Cell size < 1, adjust number of levels")

        for j in range(section**l):
            leftX = int(j * cellSizeX)
            rightX = int((j + 1) * cellSizeX)

            for i in range(section**l):
                topY = int(i * cellSizeY)
                bottomY = int((i + 1) * cellSizeY)

                GradientValueCell = GradientValue[topY:bottomY, leftX:rightX]
                GradientAngleCell = GradientAngle[topY:bottomY, leftX:rightX]

                ind = (GradientAngleCell >= angle - halfIntervalSize) | (
                    GradientAngleCell < halfIntervalSize
                )
                local_descriptor = [np.sum(GradientValueCell[ind])]

                for b in range(1, bins):
                    ind = (GradientAngleCell >= b * intervalSize - halfIntervalSize) & (
                        GradientAngleCell < (b + 1) * intervalSize - halfIntervalSize
                    )
                    local_descriptor.append(np.sum(GradientValueCell[ind]))

                descriptor.extend(local_descriptor)

    # Save descriptor as numpy array
    descriptor = np.array(descriptor)

    return descriptor


def SUMofGradient(Img):
    gradientX = np.zeros((Img.shape[0], Img.shape[1]))
    gradientY = np.zeros((Img.shape[0], Img.shape[1]))

    gradientRX, gradientRY = np.gradient(Img[:, :, 0].astype(float))
    gradientGX, gradientGY = np.gradient(Img[:, :, 1].astype(float))
    gradientBX, gradientBY = np.gradient(Img[:, :, 2].astype(float))

    gradientX = gradientRX + gradientGX + gradientBX
    gradientY = gradientRY + gradientGY + gradientBY

    return gradientX, gradientY


def maxGradient(Img):
    gradientX = np.zeros((Img.shape[0], Img.shape[1]))
    gradientY = np.zeros((Img.shape[0], Img.shape[1]))

    gradientRX, gradientRY = np.gradient(Img[:, :, 0].astype(float))
    gradientGX, gradientGY = np.gradient(Img[:, :, 1].astype(float))
    gradientBX, gradientBY = np.gradient(Img[:, :, 2].astype(float))

    for x in range(Img.shape[0]):
        for y in range(Img.shape[1]):
            maxX = gradientGX[x, y]
            if abs(gradientRX[x, y]) > abs(gradientGX[x, y]):
                maxX = gradientRX[x, y]
            if abs(gradientBX[x, y]) > abs(maxX):
                maxX = gradientBX[x, y]

            maxY = gradientGY[x, y]
            if abs(gradientRY[x, y]) > abs(gradientGY[x, y]):
                maxY = gradientRY[x, y]
            if abs(gradientBY[x, y]) > abs(maxY):
                maxY = gradientBY[x, y]

            gradientX[x, y] = maxX
            gradientY[x, y] = maxY

    return gradientX, gradientY


def compute_phog_lab(Img, bins, angle, levels, roi, section, TypeOfImage):
    # EdgeImg = edge(Img[:, :, 0], 'canny')

    if TypeOfImage == "C":
        GradientX, GradientY = maxGradient(Img)
    elif TypeOfImage == "S":
        GradientX, GradientY = SUMofGradient(Img)

    GradientValue = np.sqrt(GradientX**2 + GradientY**2)

    GradientX[GradientX == 0] = 1e-5

    YX = GradientY / GradientX
    if angle == 180:
        GradientAngle = ((np.arctan(YX) + np.pi / 2) * 180) / np.pi
    elif angle == 360:
        GradientAngle = ((np.arctan2(GradientY, GradientX) + np.pi) * 180) / np.pi

    GradientValue = GradientValue[roi[0] : roi[1], roi[2] : roi[3]]
    GradientAngle = GradientAngle[roi[0] : roi[1], roi[2] : roi[3]]
    # EdgeImg = EdgeImg[roi[0]:roi[1], roi[2]:roi[3]]
    # GradientValue = GradientValue * EdgeImg

    descriptor = computeDescriptor(GradientValue, GradientAngle, bins, angle, levels, section)

    return descriptor, GradientValue, GradientAngle


def compute_color_phog_maxlab(section, img_read, bins, angle, levels, roi=None, TypeOfImage=None):
    # check if roi is specified
    roi = roi or [0, img_read.shape[0], 0, img_read.shape[1]]

    # convert to gray image if necessary
    if len(img_read.shape) == 3:
        Img = img_read
        if Img.dtype == np.uint16:
            # divide the image by 257
            Img = Img / 257
        Img = color.rgb2lab(Img)
    else:
        Img = np.stack((img_read,) * 3, axis=-1)
        Img = color.rgb2lab(Img)

    return compute_phog_lab(Img, bins, angle, levels, roi, section, TypeOfImage)


def decolorize(picture, effect: float = 0.5, scale: Optional[float] = None, noise: float = 0.001):
    # Examine inputs
    picture = np.array(picture) / 255.0
    frame = [picture.shape[0], picture.shape[1]]
    pixels = frame[0] * frame[1]

    scale = scale or np.sqrt(2 * min(frame))

    # Reset the random number generator
    np.random.seed(0)
    tolerance = 100 * 2.2204e-16

    # Define the YPQ color space
    colorconvert = [
        [0.2989360212937753847527155, 0.5870430744511212909351327, 0.1140209042551033243121518],
        [0.5, 0.5, -1],
        [1, -1, 0],
    ]
    colorrevert = [
        [1, 0.1140209042551033243121518, 0.6440535265786729530912086],
        [1, 0.1140209042551033243121518, -0.3559464734213270469087914],
        [1, -0.8859790957448966756878482, 0.1440535265786729530912086],
    ]
    colorspan = [
        [0, 1],
        [-1, 1],
        [-1, 1],
    ]
    maxluminance = 1
    scaleluminance = 0.66856793424088827189
    maxsaturation = 1.1180339887498948482
    alter = effect * (maxluminance / maxsaturation)

    # Covert picture to the YPQ color space
    picture = picture.reshape(pixels, 3)
    image = picture.dot(colorconvert)
    original = image.copy()
    chroma = np.sqrt(image[:, 1] ** 2 + image[:, 2] ** 2)

    # Pair each pixel with a randomly chosen sample site
    mesh = np.stack(
        (
            np.tile(np.arange(1, frame[0] + 1).reshape(-1, 1), (1, frame[1])),
            np.tile(np.arange(1, frame[1] + 1), (frame[0], 1)),
        ),
        axis=2,
    ).reshape(pixels, 2)
    displace = (scale * np.sqrt(2 / np.pi)) * np.random.randn(pixels, 2)
    look = np.round(mesh + displace).astype(int)

    redo = np.where(look[:, 0] < 1)
    look[redo, 0] = 2 - np.mod(look[redo, 0], frame[0] - 1)
    redo = np.where(look[:, 1] < 2)
    look[redo, 1] = 2 - np.mod(look[redo, 1], frame[1] - 1)
    redo = np.where(look[:, 0] > frame[0])
    look[redo, 0] = frame[0] - 1 - np.mod(look[redo, 0] - 2, frame[0] - 1)
    redo = np.where(look[:, 1] > frame[1])
    look[redo, 1] = frame[1] - 1 - np.mod(look[redo, 1] - 2, frame[1] - 1)

    look = look[:, 0] + frame[0] * (look[:, 1] - 1)

    # Calculate the color differences between the paired pixels
    delta = image - image[look - 1, :]
    contrastchange = np.abs(delta[:, 0])
    contrastdirection = np.sign(delta[:, 0])
    colordifference = picture - picture[look - 1, :]
    colordifference = np.sqrt(np.sum(colordifference**2, axis=1)) + np.finfo(float).eps

    # Derive a chromatic axis from the weighted sum of chromatic differences between paired pixels
    weight = 1 - ((contrastchange / scaleluminance) / colordifference)
    weight[np.where(colordifference < tolerance)] = 0
    axis = weight * contrastdirection
    axis = delta[:, 1:3] * axis[:, None]
    axis = np.sum(axis, axis=0)

    # Project the chromatic content of the picture onto the chromatic axis
    projection = image[:, 1] * axis[0] + image[:, 2] * axis[1]
    projection = projection / (np.quantile(np.abs(projection), 1 - noise) + np.finfo(float).eps)

    # Combine the achromatic tones with the projected chromatic colors and adjust the dynamic range
    image[:, 0] = image[:, 0] + effect * projection
    imagerange = np.quantile(image[:, 0], [noise, 1 - noise])
    image[:, 0] = (image[:, 0] - imagerange[0]) / (
        imagerange[1] - imagerange[0] + np.finfo(float).eps
    )
    targetrange = effect * np.array([0, maxluminance]) + (1 - effect) * np.quantile(
        original[:, 0], [noise, 1 - noise]
    )
    image[:, 0] = targetrange[0] + image[:, 0] * (
        targetrange[1] - targetrange[0] + np.finfo(float).eps
    )
    image[:, 0] = np.minimum(
        np.maximum(image[:, 0], original[:, 0] - alter * chroma), original[:, 0] + alter * chroma
    )
    image[:, 0] = np.clip(image[:, 0], 0, maxluminance)

    # Return the results
    tones = image[:, 0] / maxluminance
    tones = tones.reshape(frame)

    # Define a function that returns multiple outputs
    def return_results():
        results = [tones]
        recolor = image.dot(colorrevert)
        recolor = np.stack(
            (
                recolor[:, 0].reshape(frame),
                recolor[:, 1].reshape(frame),
                recolor[:, 2].reshape(frame),
            ),
            axis=2,
        )
        recolor = np.clip(recolor, 0, 1)
        results.append(recolor)
        return results

    outputs = return_results()
    return outputs
