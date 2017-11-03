# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv

import numpy as np


class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order=0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order

    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns an ideal low pass mask"""
        rows, cols = shape
        x = np.linspace(-0.5, 0.5, cols) * cols
        y = np.linspace(-0.5, 0.5, rows) * rows
        radius = np.sqrt((x ** 2)[np.newaxis] + (y ** 2)[:, np.newaxis])
        mask = np.zeros(shape)
        mask[radius < cutoff] = 1
        print(radius.astype(int))

        return mask

    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask
        Hint: May be one can use the low pass filter function to get a high pass mask"""
        return 1.0 - self.get_ideal_low_pass_filter(shape, cutoff)

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        rows, cols = shape
        x = np.linspace(-0.5, 0.5, cols) * cols / 1
        y = np.linspace(-0.5, 0.5, rows) * rows / 1
        radius = np.sqrt((x ** 2)[np.newaxis] + (y ** 2)[:, np.newaxis])
        mask = 1 / (1.0 + (radius / cutoff) ** (2 * order))
        return mask

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask
        Hint: May be one can use the low pass filter function to get a high pass mask"""
        return 1. - self.get_butterworth_low_pass_filter(shape, cutoff, order)

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        rows, cols = shape
        x = np.linspace(-0.5, 0.5, cols) * cols
        y = np.linspace(-0.5, 0.5, rows) * rows
        radius = (x ** 2)[np.newaxis] + (y ** 2)[:, np.newaxis]
        mask = 1 / (2 * np.pi * cutoff ** 2) * np.exp(-1 * radius / (2 * cutoff ** 2))
        return mask

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask
        Hint: May be one can use the low pass filter function to get a high pass mask"""
        rows, cols = shape
        x = np.linspace(-0.5, 0.5, cols) * cols
        y = np.linspace(-0.5, 0.5, rows) * rows
        radius = (x ** 2)[np.newaxis] + (y ** 2)[:, np.newaxis]
        mask = 1 / (2 * np.pi * cutoff ** 2) * np.exp(-1 * radius / (2 * cutoff ** 2))
        return 1 - mask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)"""
        intensity_lower = np.amin(image)  # a
        intensity_upper = np.amax(image)  # b
        ideal_lower = 0  # c
        ideal_upper = 255  # d
        stretched = (image - ideal_lower) \
                    * ((intensity_upper - intensity_lower) / (ideal_upper - ideal_lower)) + intensity_lower
        return stretched

    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape,
            cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also
            need to take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey
            scale full contrast stretch and dtype=uint8
        """
        img_ = self.image
        forward_ = np.fft.fftshift(np.fft.fft2(img_))

        if self.order == 0:
            mask_ = self.filter(forward_.shape, self.cutoff)
        else:
            mask_ = self.filter(forward_.shape, self.cutoff, self.order)
        filtered_ = forward_ * mask_

        inverse_ = np.fft.ifft2(np.fft.ifftshift(filtered_))
        postproc_ = self.post_process_image(inverse_)
        mag_forward_ = np.log(1 + np.abs(forward_))
        mag_filtered_ = np.log(1 + np.abs(filtered_))
        return [postproc_, mag_forward_, mag_filtered_]
