#!/usr/bin/python
"""
Script supports assorted image processing capabilities
"""

### Import statements
import os, glob
import numpy as np
from numpy import pi
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import scipy.misc
import scipy.signal
import matplotlib.pyplot as plt
import math
import itertools
from PIL import Image


def _imread(image):
    """
    Handles loading of image.  Argument to function must be a valid filepath,
    PIL Image instance, or numpy array.  Returns image as float64 numpy array
    Will throw an error if any image dimensions are not even numbers as this is
    a requirement for most other functions in this script.
    """
    # If its a filepath, load from filepath
    if isinstance(image, str) and os.path.isfile(image):
        im = scipy.misc.imread(image)
    # If its a PIL image object, convert to numpy array
    elif isinstance(image, Image.Image):
        im = np.array(image)
    # If already a numpy array, just assign straight into variable
    elif isinstance(image, np.ndarray):
        im = image
    # Else, error
    else:
        raise IOError('Image must be a valid filepath, PIL Image instance, or numpy array')

    # If image is grayscale, pad a trailing dim so it works with colour pipelines
    if len(im.shape) < 3:
        im = np.expand_dims(im, axis = 2)

    # Check length and width are even numbers
    if im.shape[0] % 2 or im.shape[1] % 2:
        raise Exception('Image dimensions must be even numbers')

    # Cast to float64, return
    return im.astype(np.float64)

def _fwhm2sigma(fwhm):
    """ Converts a fwhm to a sigma, useful when setting bandwidth of filters. """
    return float(fwhm) / (2 * np.sqrt(2 * np.log(2)))

def _gaussian(X, mu, sigma):
    """
    Equation for a gaussian.
    Inputs:
      X - range of values to plot Gaussian over
      mu - mean; value to centre gaussian on
      sigma - standard deviation of gaussian
    """
    return np.exp(-0.5 * ((X-mu)/sigma)**2) / (sigma * np.sqrt(2*pi))

def _plotfftinfo(imsize):
    """
    Returns spatial frequency and orienation maps for a Fourier spectrum
    of size <imsize>,  where imsize is a (L,W) tuple or list.  Any trailing
    dimensions of imsize beyond L and W are ignored.
    """
    # Get length and width
    L = imsize[0]
    W = imsize[1]
    # Make meshgrid
    [fx, fy] = np.meshgrid(range(-W//2,W//2), range(-L//2,L//2))
    # Create maps
    fftinfo={}
    fftinfo['sf'] = ifftshift(np.sqrt(fx**2 + fy**2)) # spatial frequency
    fftinfo['ori'] = ifftshift(np.arctan2(fx,fy) % pi) # orientation
    return fftinfo


def tile(image_paths, out_path, num_rows=None, num_cols=None, by_col=False,
         base_gap=0, colgap=None, colgapfreq=None, rowgap=None, rowgapfreq=None,
         bgcol=(0,0,0,0)):
    
    # TODO: place smaller images in center of window rather than top-left
    
    # calculate num_rows and / or num_cols if not specified
    if num_rows is None and num_cols is None:
        num_rows = math.ceil(np.sqrt(len(image_paths)))
        num_cols = math.ceil(len(image_paths)/num_rows)
    elif num_rows is None:
        num_rows = math.ceil(len(image_paths)/num_cols)
    elif num_cols is None:
        num_cols = math.ceil(len(image_paths) / num_rows)

    # load images
    images = []
    for image in image_paths:
        im = Image.open(image).convert('RGBA')
        images.append(im)
    images += [None] * (num_rows * num_cols - len(images))
    
    # specify spatial arrangement of images
    order = 'F' if by_col else 'C'
    image_locations = np.arange(len(images)).reshape(
        (num_rows, num_cols), order=order)


    # width of each column is as wide as the widest image
    col_coords = []
    col_widths = []
    cumulative_width = base_gap
    for col in range(num_cols):

        col_images = [images[x] for x in image_locations[:, col]]
        widths = []
        for col_image in col_images:
            if col_image:
                widths.append(col_image.size[0])
        max_width = np.max(widths)
        col_widths.append(max_width)
        col_coords.append(cumulative_width)

        # add column gap
        cumulative_width += base_gap
        if colgap and (col+1) % colgapfreq == 0 and col < (num_cols-1):
            cumulative_width += colgap

        cumulative_width += max_width

    # height of each row is as tall as the tallest image
    row_coords = []
    row_heights = []
    cumulative_height = base_gap
    for row in range(num_rows):

        row_images = [images[x] for x in image_locations[row, :]]
        heights = []
        for row_image in row_images:
            if row_image is not None:
                heights.append(row_image.size[1])
        max_height = np.max(heights)
        row_heights.append(max_height)
        row_coords.append(cumulative_height)

        # add row gap
        cumulative_height += base_gap
        if rowgap and (row+1) % rowgapfreq == 0 and row < (num_rows - 1):
            cumulative_height += rowgap

        cumulative_height += max_height

    # build tiled image
    montage = Image.new(mode='RGBA', size=(cumulative_width, cumulative_height), color=bgcol)
    for row, col in itertools.product(range(num_rows), range(num_cols)):
        image = images[image_locations[row,col]]
        if image:
            # centre the image in the window
            width, height = image.size
            col_coord = col_coords[col] + math.floor((col_widths[col] - width)/2)
            row_coord = row_coords[row] + math.floor((row_heights[row] - height)/2)
            montage.paste(image, (col_coord, row_coord))
    montage.save(out_path)


def crop(image_path, crop_params, out_path=None):
    
    image = Image.open(image_path).convert('RGBA')
    image = image.crop(crop_params)
    if not out_path:
        out_path = image_path
    image.save(out_path)
    
    
def resize_by_dim(image_path, dim='width', new_size=256, out_path=None):
    
    image = Image.open(image_path).convert('RGBA')
    old_sizes = image.size
    if dim == 'width':
        scale = new_size / old_sizes[0]
    else:
        scale = new_size / old_sizes[1]
    new_sizes = [int(old_sizes[0] * scale), int(old_sizes[1] * scale)]
    image = image.resize(new_sizes)
    if not out_path:
        out_path = image_path
    image.save(out_path)
    
    

def center_crop_resize(image, out_path=None, image_size=[512,512]):

    if type(image) is str:
        image = Image.open(image).convert('RGBA')
    old_im_size = image.size
    min_length = min(old_im_size)
    smallest_dim = old_im_size.index(min_length)
    biggest_dim = np.setdiff1d([0,1], smallest_dim)[0]
    new_max_length = int((image_size[0]/old_im_size[smallest_dim]) * old_im_size[biggest_dim])
    new_shape = [0, 0]
    new_shape[smallest_dim] = image_size[0]
    new_shape[biggest_dim] = new_max_length
    resized_image = image.resize(new_shape)

    left = int((new_shape[0] - image_size[0]) / 2)
    right = new_shape[0] - left
    top = int((new_shape[1] - image_size[1]) / 2)
    bottom = new_shape[1] - top
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    if out_path:
        cropped_image.save(out_path)
    else:
        return cropped_image
        

def applyPhaseScram(image, coherence = 0.0, rndphi = None, mask = None):
    """
    Applies phase scrambling to grayscale or colour images.

    Parameters
    ----------
    image : any valid filepath, PIL Image instance, or numpy array
        Image to apply phase scrambling to.
    coherence : float, optional
        Number in range 0-1 that determines amount of phase scrambling to apply,
        with 0 being fully scrambled (default) and 1 being not scrambled at all.
    rndphi : array, optional
        Array of random phases same size as image.  If none provided, a
        different one will be randomly generated with each call of the function.
    mask : array, optional
        Mask of weights in range 0-1 that can be applied to rndphi (e.g. to
        scramble only certain parts of the spectrum).  Mask should be for an
        unshifted spectrum.

    Returns
    -------
    scram: ndarray
        Phase scrambled image as numpy array with uint8 datatype.

    Examples
    --------
        from imageprocessing import applyPhaseScram

        # Phase scramble image
        scram1 = applyPhaseScram('/some/image.png')

        # Scramble with 40% phase coherence
        scram2 = applyPhaseScram('/some/image.png', coherence = .4)

        # Use own random phase array
        import numpy as np
        myrndphi = np.angle(fft2(np.random.rand(im_length, im_width)))
        scram3 = applyPhaseScram('/some/image.png', rndphi = myrndphi)

        # Weight rndphi by mask.  Here we weight by an inverted horizontal-pass
        # filter to scramble vertical orientations but preserve horizontals.
        from imageprocessing import fourierFilter
        impath = '/some/image.png'
        h = fourierFilter(impath).makeOrientationFilter(45, 0, invert = True)
        scram4 = applyPhaseScram(impath, mask = h)

    """
    # Read in image
    im = _imread(image)
    L,W,D  = im.shape

    # If no random phase array specified, make one
    if rndphi is None:
        rndphi = np.angle(fft2(np.random.rand(L,W))) * (1 - coherence)

    # Weight rndphi by mask if one is provided
    if mask is not None:
        rndphi *= mask

    # Pre-allocate scrambled image
    scram = np.empty([L,W,D])
    # Loop over colour channels (for greyscale this will execute only once)
    for i in range(D):
        # Fourier transform this colour channel, calculate amplitude and phase spectra
        F = fft2(im[:,:,i])
        ampF = np.abs(F)
        phiF = np.angle(F)
        # Calculate new phase spectrum
        newphi = phiF + rndphi
        # Combine original amplitude spectrum with new phase spectrum
        newF = ampF * np.exp(newphi * 1j)
        # Inverse transform, assign into scram
        scram[:,:,i] = np.real(ifft2(newF))

    # Clip values into uint8 0-255 range
    scram[scram < 0] = 0
    scram[scram > 255] = 255

    # Squeeze trailing dims in case of grayscale, cast to uint8, return
    return np.uint8(np.squeeze(scram))




def combineAmplitudePhase(ampimage, phaseimage):
    """
    Produces composite image comprising power spectrum of powerimage and
    phase spectrum of phaseimage.  Images must be same size.

    Parameters
    ----------
    ampimage : any valid filepath, PIL Image instance, or numpy array
        Image to derive amplitude spectrum from.
    phaseimage : any valid filepath, PIL Image instance, or numpy array
        Image to derive phase spectrum from.

    Returns
    -------
    combo : ndarray
        Image derived from inputs as numpy array with uint8 datatype.
        Mean luminance is scaled to approximate the mean of the input images.

    Examples
    --------
        from imageprocessing import combineAmplitudePhase
        combo = combineAmplitudePhase('/some/image1.png', '/some/image2.png')

    """
    # Read in images
    ampim = _imread(ampimage)
    phaseim = _imread(phaseimage)

    # Check images are same shape
    if ampim.shape !=  phaseim.shape:
        raise Exception('Images must be same shape')

    # Make note of image dimensions
    L,W,D = ampim.shape

    # Pre-allocate combined image array
    combo = np.empty([L,W,D], dtype = float)

    # Loop over colour channels (for grayscale will execute only once)
    for i in range(D):
        # Get power and phase of current colour channels in relevant images
        r = np.abs(fft2(ampim[:,:,i]))
        phi = np.angle(fft2(phaseim[:,:,i]))

        # Calculate new spectrum
        z = r * np.exp(phi * 1j)

        # Inverse transform, allocate to combo
        combo[:,:,i] = np.real(ifft2(z))

    # Normalise mean luminance to mean of means of two images
    combo *= np.mean([ampim.mean(), phaseim.mean()]) / combo.mean()

    # Clip values into 0-255 uint8 range
    combo[combo < 0] = 0
    combo[combo > 255] = 255

    # Squeeze trailing dims in case of grayscale, cast to uint8, return
    return np.uint8(np.squeeze(combo))


def makeAmplitudeMask(imsize, rgb = False):
    """
    Creates an amplitude mask of specified size.

    Parameters
    ----------
    imsize : tuple or list
        Desired size of mask as (L,W) tuple or [L,W] list.  Any further trailing
        values are ignored.
    rgb : bool, optional
        If rgb = True, will create a colour mask by layering 3 amplitude masks
        into a RGB space.  Default is False.

    Returns
    -------
    ampmask : ndarray
        Requested amplitude mask as numpy array with uint8 datatype.

    Examples
    --------
        from imageprocessing import makeAmplitudeMask
        # Make 256x256 greyscale mask
        ampmask = makeAmplitudeMask([256,256])
        # Or make 256x256 colour mask
        ampmask = makeAmplitudeMask([256,256], rgb = True)

    """
    ### Sub-function definitions
    def _run(ampF):
        """
        Sub-function. Handles creation of amplitude mask
        """
        # Make random phase spectrum
        L,W = ampF.shape
        rndphi = np.angle(fft2(np.random.rand(L,W)))

        # Construct Fourier spectrum, inverse transform
        F = ampF * np.exp(1j * rndphi)
        ampmask = np.real(ifft2(F))

        # ampmask is currently centred on zero, so rescale to range 0 - 255
        ampmask -= ampmask.min() # set min to 0
        ampmask *= 255.0 / ampmask.max() # set max to 255
        return ampmask.astype(np.uint8) # cast to uint8, return

    ### Begin main function
    # Get L and W
    L = imsize[0]
    W = imsize[1]

    # Ensure dimensions are even numbers
    if L % 2 or W % 2:
        raise Exception('L and W must be even numbers')

    # Get map of the spectrum
    SFmap = _plotfftinfo([L,W])['sf']

    # Make 1/f amplitude spectrum
    with np.errstate(divide = 'ignore'): # (ignore divide by 0 warning at DC)
        ampF = 1.0 / SFmap
    ampF[0,0] = 0 # Set DC to 0 - image will be DC-zeroed

    # Make amplitude mask according to rgb argument
    if rgb:
        # If rgb requested, layer 3 amplitude masks into RGB image
        ampmask = np.empty([L,W,3], dtype = np.uint8) # pre-allocate image array
        for i in range(3):
            ampmask[:,:,i] = _run(ampF)
    else:
        # Otherwise, just make mask directly
        ampmask = _run(ampF)
    # Return
    return ampmask


def makeHybrid(image1, image2):
    """
    Makes average hybrid of images image1 and image2.

    Parameters
    ----------
    image1 : any valid filepath, PIL Image instance, or numpy array
        First image to enter into hybrid.
    image2 : any valid filepath, PIL Image instance, or numpy array
        Second image to enter into hybrid.

    Returns
    -------
    hybrid : ndarray
        Average of input images as numpy array with uint8 datatype.

    Examples
    --------
        from imageprocessing import makeHybrid
        hybrid = makeHybrid('/some/image1.png', '/some/image2.png')

    """
    # Read in images
    im1 = _imread(image1)
    im2 = _imread(image2)

    # Check images are same shape
    if im1.shape != im2.shape:
        raise Exception('Images must be same shape')

    # Define hybrid as average of two images
    hybrid = (im1 + im2) / 2.0

    # Squeeze trailing dims in case of grayscale, cast to uint8, return
    return np.uint8(np.squeeze(hybrid))


def overlayFixation(image, lum = 255, offset = 8, arm_length = 12, arm_width = 2):
    """
    Overlays fixation cross on specified image.

    Parameters
    ----------
    image : any valid filepath, PIL Image instance, or numpy array
        Image to overlay fixation cross on.
    lum : int or RGB tuple of ints, optional
        Luminance of fixation cross
    offset : int, optional
        Distance from center of the image to the nearest pixel of each arm
    arm_length : int, optional
        _length of each arm of fixation cross, specified in pixels
    arm_width : int, optional
        Thickness of each arm of fixation cross, specified in pixels (should be
        even number)

    Returns
    -------
    im : ndarray
        Image with overlaid fixaton cross as numpy array with uint8 datatype.

    Examples
    --------
        from imageprocessing import overlayFixation
        # Use default parameters
        im = overlayFixation('/some/image.png')
        # Change cross size and position, make cross black
        im2 = overlayFixation('/some/image.png', lum = 0, offset = 5,
                arm_length = 20, arm_width = 4)

    """
    # Read in image
    im = _imread(image)

    # Determine midpoint of image
    hL, hW = im.shape[0]/2, im.shape[1]/2

    # Overlay fixation cross (each arm is 2x12 pixels, extending from 8:20
    # pixels from image midpoint)
    # Left arm
    im[hL-arm_width/2 : hL+arm_width/2, hW-(arm_length+offset) : hW-offset, :] = lum
    # Right arm
    im[hL-arm_width/2 : hL+arm_width/2, hW+offset : hW+arm_length+offset, :] = lum
    # Upper arm
    im[hL-(arm_length+offset) : hL-offset, hW-arm_width/2 : hW+arm_width/2, :] = lum
    # Lower arm
    im[hL+offset : hL+arm_length+offset, hW-arm_width/2 : hW+arm_width/2, :] = lum

    # Squeeze trailing dims in case of grayscale, cast to uint8, return
    return np.uint8(np.squeeze(im))


def plotAverageAmpSpec(indir, ext = 'png', nSegs = 1, dpi = 96, cmap = 'jet'):
    """
    Calculates and plots log-scaled Fourier average amplitude spectrum
    across a number of images.

    Spectra are calculated for all images in indir with specified extension.
    Outputs are saved into a directory called "AmplitudeSpectra" created
    inside the input directory.  Outputs are (1) the average amplitude
    spectrum across images stored in a numpy array and saved as .npy file,
    and (2) contour plots of the average amplitude spectrum.

    Parameters
    ----------
    indir : str
        A valid filepath to directory containing the images to calculate
        the spectra of. All images in indir must have same dimensions.
    ext : str
        File extension of the images (default = png).
    nSegs : int, optional
        Number of segments to window image by.  Spectra are calculated within
        each window separately.  If nSegs = 1 (default), the spectrum is
        calculated across the whole image.
    dpi : int, optional
        Resolution to save plots at (default = 96).
    cmap : any valid matplotlib cmap instance
        Colourmap for filled contour plot

    Returns
    -------
    None - all outputs are saved out directly.

    Examples
    --------
        from imageprocessing import plotAverageAmpSpec
        # Calculate for whole image (leave default nSegs = 1)
        plotAverageAmpSpec('/my/images/dir', 'png')
        # Calculate within windows of image along 4x4 grid
        plotAverageAmpSpec('/my/images/dir', 'png', nSegs = 4)
        # Save at higher resolution (300dpi)
        plotAverageAmpSpec('/my/images/dir', 'png', dpi = 300)

    """
    # Ensure . character not included in extension
    ext = ext.strip('.')

    # Glob for input files
    infiles = sorted(glob.glob(os.path.join(indir, '*.%s' %ext)))
    if len(infiles) == 0:
        raise IOError('No images found! Check directory and extension')

    # Determine image dimensions from first image
    tmp = Image.open(infiles[0])
    L, W = tmp.size
    del(tmp)

    # Work out if we can segment image evenly, and dims of windows if we can
    if L % nSegs or W % nSegs:
        raise IOError('Image dimensions (%d, %d) must be divisible by nSegs (%d)' %(L, W, nSegs))
    segL = L // nSegs
    segW = W // nSegs

    # Pre-allocate array for storing spectra
    spectra = np.empty([len(infiles), L, W], dtype = float)

    # Process inputs
    print('Processing...')
    # Loop over images
    for i, infile in enumerate(infiles):
        print('\t%s' %infile)

        # Read in, grayscale (flatten) if RGB
        im = np.array(Image.open(infile).convert('L'))

        # Calculate amplitude spectrum for current window
        for y in range(0, L, segL):
            for x in range(0, W, segW):
                # Slice out window
                win = im[y:y+segL, x:x+segW]
                # Calculate amplitude spectrum for window
                ampF = np.abs(fftshift(fft2(win)))
                # Log scale, assign relevant window of spectrum (we use ampF+1
                # to avoid any -ve values from log scaling values < 1)
                spectra[i, y:y+segL, x:x+segW] = np.log(ampF + 1)
        spectra[i] /= spectra[i].max() # scale full array to range 0:1

    # Create average spectrum
    av_spectrum = spectra.mean(axis = 0)

    ### Save array, make and save plots
    print('Saving array and plots...')
    outdir = os.path.join(indir, 'AmplitudeSpectra')
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    # Main numpy array
    savename = os.path.join(outdir, 'win%d_array.npy' %nSegs)
    np.save(savename, av_spectrum)

    # Filled contour figure
    aspect_ratio = float(L) / float(W)
    figsize = (6.125, 6.125 * aspect_ratio) # 6.125 is default figure height
    fig = plt.figure(figsize = figsize)
    ax = fig.add_axes([0,0,1,1]) # add axes that fill figure
    ax.axis('off')
    ax.contour(av_spectrum, colors = 'k', origin = 'upper')
    cf = ax.contourf(av_spectrum, origin = 'upper')
    cf.set_cmap(cmap)
    cf.set_clim([0,1])
    savename = os.path.join(outdir, 'win%s_filled_contour.png' %(nSegs))
    fig.savefig(savename, dpi = dpi)
    print('Saved %s' %savename)
    plt.close(fig)

    # Line contour figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_axes([0,0,1,1]) # add axes that fill figure
    ax.axis('off')
    # Values of contour lines chosen to plot around mid-region of log-scaled spectrum in range 0:1
    ax.contour(av_spectrum, [0.45, 0.55], colors = 'k', linewidths = 2, origin = 'upper')
    savename = os.path.join(outdir, 'win%s_line_contour.png' %(nSegs))
    fig.savefig(savename, dpi = dpi)
    print('Saved %s' %savename)
    plt.close(fig)


##### CLASS DEFINITIONS #####
class FourierFilter():
    """
    Class provides functions for full pipeline of filtering images in Fourier
    domain by either spatial frequency or orientation.

    Parameters
    ----------
    image : any valid filepath, PIL Image instance, or numpy array
        Class is instantiated with image.

    Returns
    -------
    filterer : instance
        Class instance for image. Contains functions that can be used to create
        and apply filters to the image.

    Examples
    --------
        from imageprocessing import fourierFilter
        filterer = fourierFilter('/some/image.png')
        h = filterer.makeFrequencyFilter(30) # low-pass filter
        # h = filterer.makeFrequencyFilter(50, invert = True) # or high-pass
        # h = filterer.makeOrienationFilter(30, 0) # or horizontal-pass
        # h = filterer.makeOrientationFilter(30, [0, 90]) # or cardinal-pass
        filtim = filterer.applyFourierFilter(h)

    """

    def __init__(self, im):
        # Read image
        if type(im) is list:
        	self.imdims = im
        else:
        	self.im = _imread(im)
        	self.imdims = self.im.shape

    def makeFrequencyFilter(self, fwhm, mu = 0.0, invert = False):
        """
        Makes Gaussian filter of spatial frequency.

        Parameters
        ----------
        fwhm : int or float
            Full width at half maximum of filter.
        mu : int or float, optional
            Frequency to centre filter on (default = 0).
        invert : bool, optional
            Set invert = True to invert filter, e.g. to make a high-pass filter
            (default = False).

        Returns
        -------
        h : array
            Requested filter as numpy array.

        """
        L,W = self.imdims[:2]

        # Create spatial frequency map
        X = _plotfftinfo([L,W])['sf']

        # Convert fwhm to sigma, make Gaussian filter
        sigma = _fwhm2sigma(fwhm)
        h = _gaussian(X, mu, sigma)

        # Scale into range 0-1
        h /= h.max() # scale into range 0-1

        # Invert if requested
        if invert:
            h = 1 - h

        # Add in DC
        h[0,0] = 1.0

        # Return
        return h


    def makeOrientationFilter(self, fwhm, orientations, invert = False):
        """
        Makes wrapped Gaussian filter of orientation.

        Parameters
        ----------
        fwhm : int or float
            Full width at half maximum of filter.
        orientations : int or float, or tuple or list of these
            Orientation(s) to centre filter on in degrees.  If a single value,
            a single filter is created.  If multiple values are provided in a
            tuple or list, multiple filters are overlaid to allow passing
            of multiple orientations.
        invert : bool, optional
            Set invert = True to invert filer (default = False)

        Returns
        -------
        h : array
            Requested filter as numpy array.

        """
        L,W = self.imdims[:2]

        # Create orientation map
        X = _plotfftinfo([L,W])['ori']

        # Convert fwhm to sigma, then to radians
        sigma = _fwhm2sigma(fwhm)
        sigma *= pi / 180.0

        # Ensure orientations is a list (e.g. force a list if there is only 1 of them)
        if not isinstance(orientations, list):
            orientations = [orientations]

        # Convert orientations to radians
        orientations = np.array(orientations) * (pi / 180.0)

        # Pre-allocate filter
        h = np.empty([X.shape[0],X.shape[1],len(orientations)], dtype = float)

        # Loop over orientations
        for i, theta in enumerate(orientations):
            t1 = _gaussian(X, theta, sigma)
            # Wrap +/- pi
            t2 = _gaussian(X, theta+pi, sigma)
            t3 = _gaussian(X, theta-pi, sigma)
            # Sum and allocate to h
            h[:,:,i] = t1 + t2 + t3

        # Add all orientations into single filter
        h = np.sum(h, axis = 2)

        # Scale into range 0-1
        h /= h.max()

        # Invert if requested
        if invert:
            h = 1 - h

        # Add in DC
        h[0,0] = 1.0

        # Return
        return h


    def applyFourierFilter(self,h):
        """
        Apply filter to image.

        Parameters
        ----------
        h : array
            Filter to apply to image.  Can be created using functions within
            this class, or you can make your own.  Filter should be same size
            as image and for an unshifted spectrum, with values in range 0-1.

        Returns
        -------
        filtim : ndarray
            Filtered image as numpy array with uint8 datatype.

        """
        im = self.im
        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
        L,W,D = self.imdims

        # Pre-allocate filtered image
        filtim = np.empty_like(im, dtype = float)

        # Loop over colour channels (will execute only once if D == 1)
        for i in range(D):
            F = fft2(im[:,:,i]) # into frequency domain
            filtF = F * h # apply filter
            filtim[:,:,i] = np.real(ifft2(filtF)) # back to image domain
            # (take real of ifft2 to remove imaginary rounding errors)

        # Crop values into 0-255 uint8 range
        filtim[filtim < 0] = 0
        filtim[filtim > 255] = 255

        # Squeeze to remove trailing dims in case of grayscale, cast to uint8, return
        return np.uint8(np.squeeze(filtim))


class applySoftWindow():
    """
    Class provides functions for applying a cosine-ramp soft window around
    edges of image.

    Many thanks to Dan Baker for providing the original version of this script!

    Parameters
    ----------
    image : any valid filepath, PIL Image instance, or numpy array
        Class is instantiated with image.

    Returns
    -------
    windower: instance
        Class instance for image.  Contains functions that can be used to
        create and apply a soft-window mask to an image.

    Examples
    --------
    from imageprocessing import applySoftWindow
    windower = applySoftWindow('/some/image.png')

    # Create rectangular mask
    mask = windower.createMask('rect')
    # Apply mask
    winIm = windower.applyMask(mask)

    # If you have more images to run and they have the same dimensions you
    # can speed up computation time by re-using the original mask
    winIm2 = applySoftWindow('/some/other/image.png').applyMask(mask)

    # Create an elliptical mask with a fwhm of 0.8
    mask2 = windower.createMask('ellipse', fwhm = 0.8)
    # Apply to image, set background to be white
    winIm3 = windower.applyMask(mask2, bglum = 255)

    """
    def __init__(self, image):
        # Read in image
        self.im = _imread(image)
        self.imshape = self.im.shape


    def createMask(self, maskshape, fwhm = 0.9):
        """
        Create soft-window mask.

        Parameters
        ----------
        maskshape : {'ellipse', 'rect'}
            Desired shape of mask (elliptical or rectangular)
        fwhm : float, optional
            Value in range 0-1; dimensions of the mask as a proportion of the
            dimensions of the image (default = 0.9)

        Returns
        -------
        mask : ndarray
            Mask as numpy array with datatype float.

        """
        # Extract image length and width for brevity
        L, W = self.imshape[:2]

        ### Create cosine smoothing kernel
        # Determine width of blur along horizontal / vertical edges of mask
        x_blurW = int(W * (1 - fwhm))
        y_blurW = int(L * (1 - fwhm))
        # Make cosine half cycles for horizontal / vertical dims
        x_cosine = np.matrix(np.cos(np.linspace(-pi/2.0, pi/2.0, x_blurW)))
        y_cosine = np.matrix(np.cos(np.linspace(-pi/2.0, pi/2.0, y_blurW)))
        # Combine into 2D smoothing kernel
        winKernel = np.dot(y_cosine.T, x_cosine)

        ### Create mask for requested shape
        # Start with ideal mask (ones inside mask, zeros outside)
        if maskshape == 'ellipse':
            # Work out radii of x and y dims
            x_radius = W * fwhm / 2.0
            y_radius = L * fwhm / 2.0
            # Make mask
            [fx,fy] = np.meshgrid(range(W),range(L))
            mask = (((fx - W/2.0)/x_radius)**2 + ((fy-L/2)/y_radius)**2 < 1).astype(float)
        elif maskshape == 'rect':
            # Initialise blank mask
            mask = np.zeros([L,W])
            # Work out border width along x and y dims
            x_bordW = int(x_blurW/2)
            y_bordW = int(y_blurW/2)
            # Fill in mask
            mask[y_bordW:-(y_bordW+1), x_bordW:-(x_bordW+1)] = 1
        else:
            raise ValueError('Unrecognised argument to \'maskshape\'')

        # Convolve mask with winKernel to give blurred edge to mask
        # (We do an fftconvolve as these are hugely faster and have a much smaller
        # memory load than direct convolutions when dealing with large arrays such
        # as images)
        mask = scipy.signal.fftconvolve(mask, winKernel, mode = 'same')

        # Rescale to max of 1
        mask /= mask.max()

        # Return
        return mask


    def applyMask(self, mask, bglum  = 'mean'):
        """
        Apply mask to image.

        Parameters
        ----------
        mask : ndarray
            Mask to be applied to image, e.g. one returned by .createMask()
            method
        bglum : {'mean', int or float, RGB tuple of ints or floats}, optional
            Luminance to set background outside masked region to.
            If set to 'mean' (default) the mean image luminance is used.

        Returns
        -------
        im : ndarray
            Masked image as numpy array with datatype uint8.

        """
        # Extract image and image depth for brevity
        im = self.im
        D = self.imshape[2]

        # If no bglum provided, use mean luminance of image
        if bglum == 'mean':
            bglum = im.mean()

        # Subtract bglum
        im -= bglum

        # Apply mask to each colour channel in turn (will execute only once if D == 1)
        for i in range(D):
            im[:,:,i] *= mask

        # Add bglum back in
        im += bglum

        # Squeeze any trailing dims in case of grayscale, cast to uint8, return
        return np.uint8(np.squeeze(im))




# If someone erroneously attempts to run this script from the commandline,
# inform them they should be importing it from within python instead
if __name__ == '__main__':
    print(
"""
This script is a python module - functions and classes should be imported from
within python (instead of running the script from the commandline).

Try the following:

cd /directory/containing/this/script/
ipython
# (wait for ipython to start)
import imageprocessing
help(imageprocessing)
"""
    )
