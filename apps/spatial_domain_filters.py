from PIL import Image
from skimage import filters, io
import streamlit as st
from skimage.color import rgba2rgb, rgb2gray
import numpy as np

def_lowpass_image = "./images/boy.webp"


def laplacianFilter(st):
    st.header("Laplacian Filter")
    original_image = st.file_uploader("")
    if original_image is None:
        original_image = def_lowpass_image
    original_image = Image.open(original_image)
    original_image = np.array(original_image)
    
    # If the image is RGBA, convert it to RGB
    if original_image.shape[-1] == 4:
        original_image = rgba2rgb(original_image)
    
    kSize = st.slider("Kernal Size", min_value=3, max_value=10, step=1, value=3)
    
    gray_image = rgb2gray(original_image)
    laplacian_image = filters.laplace(gray_image, ksize=kSize)

    laplacian_image = np.clip(laplacian_image, 0,1)

    # Display results
    col1, col2 = st.columns(2)
    col1.image(original_image, use_column_width=True, caption="Original Image")
    col2.image(laplacian_image, use_column_width=True, caption="Laplacian Filtered Image")


def sobelFilter(st):
    st.header("Sobel Filter")

    original_image = st.file_uploader("")
    if original_image is None:
        original_image = def_lowpass_image
    original_image = Image.open(original_image)
    original_image = np.array(original_image)
    
    # If the image is RGBA, convert it to RGB
    if original_image.shape[-1] == 4:
        original_image = rgba2rgb(original_image)

    # Convert to grayscale as Sobel operator requires a single channel image
    gray_image = rgb2gray(original_image)
    
    # Apply Sobel filter
    edge_sobel_h = filters.sobel_h(gray_image)
    edge_sobel_v = filters.sobel_v(gray_image)
    edge_sobel = np.sqrt(edge_sobel_h**2 + edge_sobel_v**2)

    edge_sobel_v = np.clip(edge_sobel_v, 0, 1)
    edge_sobel_h = np.clip(edge_sobel_h, 0, 1)
    edge_sobel = np.clip(edge_sobel, 0, 1)

    col1, col2 = st.columns(2)

    col1.image(original_image, caption="Original Image")
    col2.image(edge_sobel, caption="Resultant Sobel Image")
    col1.image(edge_sobel_h,caption="Image After Horizontal Edge Detection")
    col2.image(edge_sobel_v, caption="Image After Vertical Edge Detection")

def lowpassGaussianFilter(st):
    st.header("Lowpass Gaussian Filter")
    original_image = st.file_uploader("")
    if original_image is None:
        original_image = def_lowpass_image
    original_image = Image.open(original_image)
    original_image = np.array(original_image)
    
    # If the image is RGBA, convert it to RGB
    if original_image.shape[-1] == 4:
        original_image = rgba2rgb(original_image)

    sigma = st.slider("Sigma", min_value=0.1, max_value=20.0, step=0.1, value=3.7)
    
    filtered_image = filters.gaussian(original_image, sigma=sigma, channel_axis=3)
    filtered_image = np.clip(filtered_image, 0, 1)

    org_img_col, out_img_col = st.columns(2)
    org_img_col.image(original_image, use_column_width=True, caption="Original Image")
    out_img_col.image(filtered_image, use_column_width=True, caption="Lowpass Filtered Image")


def highpassGaussianFilter(st):
    st.header("Highpass Gaussian Filter")
    original_image = st.file_uploader("")
    if original_image is None:
        original_image = def_lowpass_image
    original_image = Image.open(original_image)
    original_image = np.array(original_image)

    # If the image is RGBA, convert it to RGB
    if original_image.shape[-1] == 4:
        original_image = rgba2rgb(original_image)

    sigma = st.slider("Sigma", min_value=0.1, max_value=20.0, step=0.1, value=3.7)
    threshold_value = st.slider("Threshold", min_value=0.001, max_value=1.0, step=0.001, value=0.5)

    lowpass_filtered_image = filters.gaussian(original_image, sigma=sigma, channel_axis=3)
    lowpass_filtered_image = np.clip(lowpass_filtered_image, 0, 1)

    highpass_filtered_image = original_image - lowpass_filtered_image
    highpass_filtered_image = np.clip(highpass_filtered_image, 0, 1)

    grayscale_highpass_image = rgb2gray(highpass_filtered_image)
    
    binary_highpass_image = grayscale_highpass_image < threshold_value
    binary_highpass_image = binary_highpass_image.astype(float)

    org_img_col, highpass_img_col = st.columns(2)
    org_img_col.image(original_image, use_column_width=True, caption="Original Image")
    highpass_img_col.image(binary_highpass_image, use_column_width=True, caption="Highpass Filtered Image")

spatial_filters = {
    "Laplacian Filter": laplacianFilter,
    "Sobel Filter": sobelFilter,
    "Lowpass Gaussian Filter": lowpassGaussianFilter,
    "Highpass Laplacian Filter": highpassGaussianFilter,
}


def SpatialDomainFilter(st):

    content_col, filter_col = st.columns([2, 1])

    if "spatical_domain_filter" not in st.session_state:
        st.session_state.spatical_domain_filter = "Laplacian Filter"

    with filter_col:
        st.header("Filters")
        for filter_name in spatial_filters:
            if st.button(filter_name):
                st.session_state.spatical_domain_filter = filter_name

    with content_col:
        if 'spatical_domain_filter' in st.session_state and st.session_state.spatical_domain_filter:
            filter_func = spatial_filters[st.session_state.spatical_domain_filter]
            filter_func(content_col)
        else:
            st.write("Select a filter from the right column to see results here.")

