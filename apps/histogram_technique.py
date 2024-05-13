from PIL import Image
import numpy as np
from skimage import exposure, data

def_image = "./images/cat_image_1.jpeg"
def_ref = "./images/car_d.jpeg"

def_image_for_hm = data.chelsea()
def_ref_for_hm = data.coffee()

def_image_for_he = data.moon()

def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = (image - image_min) / (image_max - image_min)
    return normalized_image

def histogramEquilizer(st):
    st.header("Histogram Equilizer")
    original_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if original_image is not None:
        image = Image.open(original_image)
    else:
        image = def_image_for_he
    img_array = np.array(image)
    output = exposure.equalize_hist(img_array)

    col1, col2 =  st.columns(2)
    col1.image(image, caption="Original Image",use_column_width=True)
    col2.image(output, caption="Resultant Image", use_column_width=True)

def histogramMatching(st):
    st.header("Histogram Matching")
    original_image = st.file_uploader("Original Image", type=["png", "jpg", "jpeg"])
    
    reference_image = st.file_uploader("Reference Image")
    
    if original_image is not None:
        original_image = Image.open(original_image)
    else:
        original_image = def_image_for_hm

    if reference_image is not None:
        reference_image = Image.open(reference_image)
    else:
        reference_image = def_ref_for_hm
    original_array = np.array(original_image,dtype=np.float64)
    reference_array = np.array(reference_image,dtype=np.float64)

    resultant_image = exposure.match_histograms(original_array, reference_array)

    col1, col2 = st.columns(2)

    col1.image(original_image, "Original Image", use_column_width=True)
    col2.image(reference_image, "Reference Image", use_column_width=True)

    resultant_image = normalize_image(resultant_image)
    st.image(resultant_image, "Resultant Image", use_column_width=True)

histogram_techniques = {
    "Histogram Equilizer": histogramEquilizer,
    "Histogram Matching": histogramMatching,
}

def HistogramTechnique(st):
    content_col, technique_col = st.columns([2,1])

    if "histogram_techniques" not in st.session_state:
        st.session_state.histogram_techniques = "Histogram Equilizer"

    with technique_col:
        st.header("Techniques")
        for technique_name in histogram_techniques:
            if st.button(technique_name):
                st.session_state.histogram_techniques = technique_name
    
    with content_col:
        if 'histogram_techniques' in st.session_state and st.session_state.histogram_techniques:  
            technique_func = histogram_techniques[st.session_state.histogram_techniques]
            technique_func(content_col)
        else:
            st.write("Select a technique from the right column to see results here.")
