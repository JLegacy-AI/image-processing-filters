import cv2
import numpy as np
from PIL import Image

def_image = "./images/cat_image.jpeg"

def bandpassFilter(st):
    st.header("Bandpass Filter")

    original_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    original_image = def_image if original_image is None else original_image

    # Open the image with PIL and convert to NumPy array
    original_image_color = Image.open(original_image)
    original_image = Image.open(original_image).convert('L')  # Convert image to grayscale
    original_image = np.array(original_image)

    # Get user inputs for frequency cut-offs
    low_freq = st.slider("Low Frequency Cut-off", 1, 100, 80)
    high_freq = st.slider("High Frequency Cut-off", 1, 100, 5)
    
    # FFT to convert the image to frequency domain
    dft = cv2.dft(np.float32(original_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = original_image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with dimensions appropriate for dft_shift
    mask = np.zeros((rows, cols, 2), np.uint8)  # Mask must be the same size as the DFT output
    mask[crow-low_freq:crow+low_freq, ccol-low_freq:ccol+low_freq] = 1
    mask[crow-high_freq:crow+high_freq, ccol-high_freq:ccol+high_freq] = 0

    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    # Normalize the image back to the range [0, 255]
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    col1, col2 = st.columns(2)
    col1.image(original_image_color, "Original Image", use_column_width=True)
    col2.image(img_back, "Lowpass Filtered Image", use_column_width=True)

def butterworthFilter(st):
    st.header("Butterworth Filter")
    original_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    original_image = def_image if original_image is None else original_image

    # Open the image with PIL and convert to NumPy array
    original_image_color = Image.open(original_image)
    original_image = Image.open(original_image).convert('L')  # Convert image to grayscale
    original_image = np.array(original_image)

    # Get user input for the cutoff frequency and the order of the filter
    cutoff = st.slider("Cutoff Frequency", min_value=10, max_value=100, value=30)
    order = st.slider("Order of the Filter", min_value=1, max_value=10, value=2)
    
    # FFT to convert the image to the frequency domain
    dft = np.fft.fft2(original_image)
    dft_shift = np.fft.fftshift(dft)

    # Create Butterworth filter
    rows, cols = original_image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    filter_mask = 1 / (1 + (radius / cutoff)**(2 * order))

    # Apply the filter
    filtered_dft_shift = dft_shift * filter_mask

    # Inverse FFT to convert back to the spatial domain
    f_ishift = np.fft.ifftshift(filtered_dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize the image to 0-255 and convert to uint8
    img_back = np.interp(img_back, (img_back.min(), img_back.max()), (0, 255)).astype(np.uint8)
    
    col1, col2 = st.columns(2)
    col1.image(original_image_color, "Original Image", use_column_width=True)
    col2.image(img_back, "Lowpass Filtered Image", use_column_width=True)


def highpassFilter(st):
    st.header("Highpass Filter")
    original_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    original_image = def_image if original_image is None else original_image
    cutoff = st.slider("Cutoff Frequency", min_value=10, max_value=100, value=30)

    # Open the image with PIL and convert to NumPy array
    original_image_color = Image.open(original_image)
    original_image = Image.open(original_image).convert('L')  # Convert image to grayscale
    original_image = np.array(original_image)

    # FFT to convert the image to the frequency domain
    dft = np.fft.fft2(original_image)
    dft_shift = np.fft.fftshift(dft)

    # Create a highpass filter
    rows, cols = original_image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    highpass_filter = radius > cutoff  # Create a highpass filter

    # Apply the filter
    filtered_dft_shift = dft_shift * highpass_filter

    # Inverse FFT to convert back to the spatial domain
    f_ishift = np.fft.ifftshift(filtered_dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize the image to 0-255 and convert to uint8
    img_back = np.interp(img_back, (img_back.min(), img_back.max()), (0, 255)).astype(np.uint8)
    
    col1, col2 = st.columns(2)
    col1.image(original_image_color, "Original Image", use_column_width=True)
    col2.image(img_back, "Lowpass Filtered Image", use_column_width=True)

def lowpassFilter(st):
    st.header("Lowpass Filter")
    original_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    original_image = def_image if original_image is None else original_image
    cutoff = st.slider("Cutoff Frequency", min_value=10, max_value=100, value=30)

    # Open the image with PIL and convert to NumPy array
    original_image_color = Image.open(original_image)
    original_image = Image.open(original_image).convert('L')  # Convert image to grayscale
    original_image = np.array(original_image)

    # FFT to convert the image to the frequency domain
    dft = np.fft.fft2(original_image)
    dft_shift = np.fft.fftshift(dft)

    # Create a lowpass filter
    rows, cols = original_image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    lowpass_filter = radius < cutoff  # Create a lowpass filter

    # Apply the filter
    filtered_dft_shift = dft_shift * lowpass_filter

    # Inverse FFT to convert back to the spatial domain
    f_ishift = np.fft.ifftshift(filtered_dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize the image to 0-255 and convert to uint8
    img_back = np.interp(img_back, (img_back.min(), img_back.max()), (0, 255)).astype(np.uint8)
    

    col1, col2 = st.columns(2)
    col1.image(original_image_color, "Original Image", use_column_width=True)
    col2.image(img_back, "Lowpass Filtered Image", use_column_width=True)

frequency_filters = {
    "Bandpass Filter": bandpassFilter,
    "Butterworth Filter": butterworthFilter,
    "Highpass Filter": highpassFilter,
    "Lowpass Filter": lowpassFilter,
}


def FrequencyDomainFilters(st):
    content_col, filter_col = st.columns([2,1])

    if "frequency_domain_filter" not in st.session_state:
        st.session_state.frequency_domain_filter = "Bandpass Filter"
    
    with filter_col:
        st.header("Filters")
        for filter_name in frequency_filters:
            if st.button(filter_name):
                st.session_state.frequency_domain_filter = filter_name
    
    with content_col:
        if 'frequency_domain_filter' in st.session_state and st.session_state.frequency_domain_filter:
            
            filter_func = frequency_filters[st.session_state.frequency_domain_filter]
            filter_func(st)

        else:
            st.write("Select a filter from the right column to see results here.")