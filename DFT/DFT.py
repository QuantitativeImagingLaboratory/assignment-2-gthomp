# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import copy
import numpy as np


class DFT:
    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        img_ = copy.deepcopy(matrix)
        rows = img_.shape[0]
        cols = img_.shape[1]

        fft_ = np.zeros(matrix.shape, dtype=complex)
        for k in range(rows):
            for l in range(cols):
                outer_sum_ = 0
                for n in range(cols):
                    inner_sum_ = 0
                    for m in range(rows):
                        inner_sum_ += img_[m, n] * np.exp(-2.0j * np.pi * m * k / rows)
                    outer_sum_ += inner_sum_ * np.exp(-2.0j * np.pi * n * l / cols)
                fft_[k, l] = outer_sum_

        return fft_

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        fft_ = copy.deepcopy(matrix)
        rows = fft_.shape[0]
        cols = fft_.shape[1]

        img_ = np.zeros(matrix.shape, dtype=complex)
        for m in range(rows):
            for n in range(cols):
                outer_sum_ = 0
                for l in range(cols):
                    inner_sum_ = 0
                    for k in range(rows):
                        inner_sum_ += fft_[k, l] * np.exp(2.0j * np.pi * k * m / rows)
                    outer_sum_ += inner_sum_ * np.exp(2.0j * np.pi * l * n / cols)
                img_[m, n] = outer_sum_

        return img_

    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""

        return matrix

    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        return np.log(1 + np.abs(matrix))
