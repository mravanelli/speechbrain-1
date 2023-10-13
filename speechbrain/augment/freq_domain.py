"""Frequency-Domain Sequential Data Augmentation Classes

This module comprises classes tailored for augmenting sequential data in the
frequency domain, such as spectrograms and mel spectrograms.
Its primary purpose is to enhance the resilience of neural models during the training process.

Authors:
- Peter Plantinga (2020)
- Mirco Ravanelli (2023)
"""

import torch


class SpectrogramDrop(torch.nn.Module):
    """This class drops slices of the input spectrogram.

    Using `SpectrogramDrop` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.

    Reference:
        https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to drop the
        spectrogram, in samples.
    drop_length_high : int
        The high end of lengths for which to drop the
        signal, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped.
    replace_with_zero: bool.
        If true the masked values are replaced with zeros.
        Otherwise, with the mean value of the spectrogram.
    dim : int
        Corresponding dimension to mask. If dim=1, we apply time masking.
        If dim=2, we apply frequency masking.

    Example
    -------
    >>> # time-masking
    >>> drop = SpectrogramDrop(dim=1)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = drop(spectrogram)
    torch.Size([4, 150, 40])
    >>> # frequency-masking
    >>> drop = SpectrogramDrop(dim=2)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = drop(spectrogram)
    torch.Size([4, 150, 40])
    """

    def __init__(
        self,
        drop_length_low=5,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace_with_zero=True,
        dim=1,
    ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.replace_with_zero = replace_with_zero
        self.dim = dim

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

    def forward(self, spectrogram):
        """
        Apply the DropChunk augmentation to the input spectrogram.

        This method randomly drops chunks of the input spectrogram to augment the data.

        Arguments
        ---------
        spectrogram : torch.Tensor
            Input spectrogram of shape `[batch, time, fea]`.

        Returns
        -------
        torch.Tensor
            Augmented spectrogram of shape `[batch, time, fea]`.
        """

        # Manage 4D tensors
        if spectrogram.dim() == 4:
            spectrogram = spectrogram.view(
                -1, spectrogram.shape[2], spectrogram.shape[3]
            )

        # Get the batch size
        batch_size, time_duration, fea_size = spectrogram.shape

        # Managing masking dimensions
        if self.dim == 1:
            D = time_duration
        else:
            D = fea_size

        # Randomly select the number of chunks to drop (same for all samples in the batch)
        n_masks = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(1,),
            device=spectrogram.device,
        )

        # Randomly sample the lengths of the chunks to drop
        mask_len = torch.randint(
            low=self.drop_length_low,
            high=self.drop_length_high,
            size=(batch_size, n_masks),
            device=spectrogram.device,
        ).unsqueeze(2)

        # Randomly sample the positions of the chunks to drop
        mask_pos = torch.randint(
            0,
            max(1, D, -mask_len.max()),
            (batch_size, n_masks),
            device=spectrogram.device,
        ).unsqueeze(2)

        # Compute the mask for the selected chunk positions
        arange = torch.arange(D, device=spectrogram.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        mask = mask.unsqueeze(2) if self.dim == 1 else mask.unsqueeze(1)

        # Determine the value to replace the masked chunks (zero or mean of the spectrogram)
        val = 0.0 if self.replace_with_zero else spectrogram.mean().detach()

        # Apply the mask to the spectrogram
        spectrogram = spectrogram.masked_fill_(mask, val)

        return spectrogram.view(*spectrogram.shape)


class Warping(torch.nn.Module):
    """
    Apply time or frequency warping to a spectrogram.

    If `dim=1`, time warping is applied; if `dim=2`, frequency warping is applied.
    This implementation selects a center and a window length to perform warping.
    It ensures that the temporal dimension remains unchanged by upsampling or
    downsampling the affected regions accordingly.

    Reference:
        https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    warp_window : int, optional
        The width of the warping window. Default is 5.
    warp_mode : str, optional
        The interpolation mode for time warping. Default is "bicubic."
    dim : int, optional
        Dimension along which to apply warping (1 for time, 2 for frequency).
        Default is 1.

    Example
    -------
    >>> # Time-warping
    >>> warp = Warping()
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram.shape)
    torch.Size([4, 150, 40])
    >>> out = warp(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    >>> # Frequency-warping
    >>> warp = Warping(dim=2)
    >>> spectrogram = torch.rand(4, 150, 40)
    >>> print(spectrogram shape)
    torch.Size([4, 150, 40])
    >>> out = warp(spectrogram)
    >>> print(out.shape)
    torch.Size([4, 150, 40])
    """

    def __init__(self, warp_window=5, warp_mode="bicubic", dim=1):
        super().__init__()
        self.warp_window = warp_window
        self.warp_mode = warp_mode
        self.dim = dim

    def forward(self, spectrogram):
        """
        Apply warping to the input spectrogram.

        Arguments
        ---------
        spectrogram : torch.Tensor
            Input spectrogram with shape `[batch, time, fea]`.

        Returns
        -------
        torch.Tensor
            Augmented spectrogram with shape `[batch, time, fea]`.
        """

        # Set warping dimension
        if self.dim == 2:
            spectrogram = spectrogram.transpose(1, 2)

        original_size = spectrogram.shape
        window = self.warp_window

        # 2d interpolation requires 4D or higher dimension tensors
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(1)

        len_original = spectrogram.shape[2]
        if len_original - window <= window:
            return spectrogram.view(*original_size)

        # Compute center and corresponding window
        c = torch.randint(window, len_original - window, (1,))[0]
        w = torch.randint(c - window, c + window, (1,))[0] + 1

        # Update the left part of the spectrogram
        left = torch.nn.functional.interpolate(
            spectrogram[:, :, :c],
            (w, spectrogram.shape[3]),
            mode=self.warp_mode,
            align_corners=True,
        )

        # Update the right part of the spectrogram.
        # When the left part is expanded, the right part is compressed by the 
        # same factor, and vice versa.
        right = torch.nn.functional.interpolate(
            spectrogram[:, :, c:],
            (len_original - w, spectrogram.shape[3]),
            mode=self.warp_mode,
            align_corners=True,
        )

        # Injecting the warped left and right parts.
        spectrogram[:, :, :w] = left
        spectrogram[:, :, w:] = right
        spectrogram = spectrogram.view(*original_size)

        # Transpose if freq warping is applied.
        if self.dim == 2:
            spectrogram = spectrogram.transpose(1, 2)

        return spectrogram