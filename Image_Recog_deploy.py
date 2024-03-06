import streamlit as st
from stardist.models import StarDist2D
from stardist import random_label_cmap
from pyvis.network import Network
from scipy.ndimage import label
import cv2
import imutils#
import networkx as nx
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import math
from scipy.ndimage import label, find_objects
import plotly.figure_factory as ff
from collections import Counter
import gc
# from streamlit_profiler import Profiler
# from memory_profiler import profile




# Function to process and convert image to .tif and resize if necessary
#@st.cache_data
#@profile
def process_image(uploaded_image):
    # Check if an image is uploaded
    if uploaded_image:
        # Create a temporary in-memory buffer to save the image as .tif
        image = Image.open(uploaded_image)
        tif_buffer = BytesIO()
        image.save(tif_buffer, format="TIFF")
        img = Image.open(tif_buffer)
        img = np.array(img)
        if len(img.shape) == 3 and img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Normalize the pixel values to the range [0, 1]
        min_value = img.min()
        max_value = img.max()
        img = (img - min_value) / (max_value - min_value)
        height, width = img.shape
        max_dim = max(height, width)
        if max_dim > 1024:
            scale = 1024 / max_dim
            new_height = int(height * scale)
            new_width = int(width * scale)
            img = cv2.resize(img, (new_width, new_height))
        return img  # Return the temporary file path
#@st.cache_data
#@profile
def find_scale(selected_template, img, template1, template2):
    img = img.astype(np.float32)
    # Select the appropriate template
    if selected_template == "Template 1":
        template = template1
    else:
        template = template2
    # Convert the data type of the template image to match the data type of the input image (8-bit absolute values)
    template = template.astype(np.uint8)
    # Detect edges in the template image using Canny edge detector
    template_edges = cv2.Canny(template, 50, 150)
    template_edges = template_edges.astype(np.float32)
    # Initialize the best match values
    best_match = None
    best_scale = None
    best_loc = None
    #iteratively resize the template to match it to the image in 1% percent steps
    for scale in range(5, 100, 1):
        # Resize the template image to the current scale
        resized_template = imutils.resize(template_edges, width=int(template_edges.shape[1] * scale / 100))
        resized_template = resized_template.astype(np.float32)
        # If the resized template is larger than the input image, break the loop
        if resized_template.shape[0] > img.shape[0] or resized_template.shape[1] > img.shape[1]:
            pass
        else:
            # img = img.astype(np.uint8)
            img_conv = cv2.convertScaleAbs(img, alpha=(255.0 / 1.0))
            image_edge = cv2.Canny(img_conv, 50, 150)
            image_edge = image_edge.astype(np.float32)
            # Perform template matching using cv2.TM_CCOEFF method
            result = cv2.matchTemplate(image_edge, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            # Update the best match values if a better match is found
            if best_match is None or max_val > best_match:
                best_match = max_val
                best_scale = scale / 100
                best_loc = max_loc
                best_template = resized_template
                best_scale = scale
    # Get the location of the scale bar
    top_left = (int(best_loc[0]), int(best_loc[1]))
    bottom_right = (int(best_loc[0] + best_template.shape[1]), int(best_loc[1] + best_template.shape[0]))
    # st.write(top_left,bottom_right)
    # st.write("image after match template (should not be any difference)")
    # Draw a rectangle around the scale bar
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # st.write("after to_color before rectangle",img.shape)
    # put rectangle around scale on the image
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
    # Calculate the length of the scale bar
    length = bottom_right[0] - top_left[0]
    img.astype(np.float32)
    # Display the result
    st.image(img, caption='Detected scale bar', use_column_width=True, clamp=True)
    st.write(f'The length of the scale bar is {length} pixels.')
    return length



@st.cache_resource
#@profile
def get_model():
    # if model_change == 'Basic':
    #     model = StarDist2D.from_pretrained('2D_versatile_fluo')
    # elif model_change == 'Fine_Tuned':
    #     model = StarDist2D(None, name = "FineTuned_v3", basedir='Models') # loading model
    # elif model_change == 'Self_Trained':
    model = StarDist2D(None, name = "Self_Trained", basedir='Models') # loading model
    return model

#@st.cache_data
# @st.cache_resource
#@profile
def stardist(file, PBS, NMS):
    st.cache_resource.clear()

    model = get_model()
    # try:
    st.session_state['labels'], st.session_state['details'] = model.predict_instances(file, prob_thresh=PBS, nms_thresh=NMS) # predicting masks

# @st.cache_resource
#@profile
def display_prediction(L_scale, scale):
    # find Contours
    contour_x = []
    contour_y = []
    for contour in st.session_state['details']['coord']:
        contour_x.append(contour[0])
        contour_y.append(contour[1])
    # find centerpoints
    center_x = []
    center_y = []
    for coord in st.session_state['details']['points']:
        center_x.append(coord[0])
        center_y.append((coord[1]))
    # identify number of unique objects
    num_objects = len(st.session_state['details']['points'])
    # Calculate the number of pixels in each object and convert into actual area, using scale
    object_sizes = [round(len(np.where(st.session_state['labels'] == i)[0])*((scale/L_scale)**2),3) for i in range(1, num_objects)]
    # Create the overlaid image using Matplotlib with adjusted figure size
    cmap = random_label_cmap()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(preprocessed_image, clim=(0, 1), cmap='gray')
    ax.scatter(center_y, center_x, c='red', s=5)
    ax.scatter(contour_y, contour_x, c='red', s=3)
    ax.imshow(st.session_state['labels'], cmap=cmap, alpha=0.5)
    ax.axis('off')
    # Specify a fixed size for the Matplotlib figure in col2
    st.pyplot(fig, use_container_width=True, clear_figure=True)
    return object_sizes, num_objects
#@st.cache_data
#@profile
def display_selected_labels(adjusted_object_sizes, toggle):
    cmap = random_label_cmap()
    labels_int = [indice for value, indice in adjusted_object_sizes]
    # Create a labeled objects array with only the selected objects within the specified size range
    filtered_labeled_objects = np.where(np.isin(st.session_state['labels'], labels_int),
                                        st.session_state['labels'], 0)
    # plot the labels selected by size
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(preprocessed_image, clim=(0, 1), cmap='gray')
    ax.imshow(filtered_labeled_objects, cmap=cmap, alpha=0.5)
    # get coordinates and contours of selected samples
    x_center_select = []
    y_center_select = []
    x_contour_select = []
    y_contour_select = []
    text_select = []
    for i in labels_int:
        # filtered_labeled_objects[labels == idx] = i
        x_center_select.append(st.session_state['details']['points'][i-1][1])
        y_center_select.append(st.session_state['details']['points'][i-1][0])
        x_contour_select.append(st.session_state['details']['coord'][i-1][1])
        y_contour_select.append(st.session_state['details']['coord'][i-1][0])
        #iterating through adjusted_object_sizes to find the corresponding object size
        for value, indice in adjusted_object_sizes:
            if indice == i:
                text_select.append(value)
    object_diameters = []
    object_diameters_average = []
    # iterating through the centerpoints and contours of the already filtered label data
    for i, (x_center, y_center) in enumerate(zip(x_center_select, y_center_select)):
        diameters = []
        for j, (x_contour, y_contour) in enumerate(zip(x_contour_select[i], y_contour_select[i])):
            diameter = 2 * calculate_distance(x_center, y_center, x_contour, y_contour)
            diameters.append(1 / (st.session_state['scale_length'] / (st.session_state['mikro_scale'] * diameter)))
        object_diameters.append(diameters)
        object_diameters_average.append(round(np.average(diameters),3))
    if toggle:
        text_output = text_select
    else:
        text_output = object_diameters_average
    # plot object sizes ontop of objects
    for x, y, text in zip(x_center_select, y_center_select, text_output):
        ax.text(x, y, text, fontsize=10, color='red')
    ax.axis('off')
    # Display the figure in Streamlit
    st.pyplot(fig, use_container_width=True, clear_figure=True)
    return x_center_select, y_center_select, x_contour_select, y_contour_select, filtered_labeled_objects
#@st.cache_data
#@profile
def plot_hist(object_sizes, num_objects):
    object_sizes = [size for size, i in object_sizes]
    area_range = max(object_sizes) - min(object_sizes)
    hist_data = [object_sizes]
    # Create distplot with custom bin_size
    if 'bins' not in st.session_state:
        bins = num_objects / 3
    else:
        # bin_size = area_range / st.session_state['bins']
        bins = st.session_state['bins']
    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 2))
    # Add the histogram in the subplot
    ax.hist(object_sizes, bins=bins, edgecolor='k')
    ax.set_xlabel('Object Size in µm^2')
    ax.set_ylabel('Occurence')
    ax.set_title('Histogram of Object Sizes')
    # Show the entire figure
    plt.tight_layout()
    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig, use_container_width=True, clear_figure=True)
#@st.cache_data
#@profile
def calculate_distance(x1, y1, x2, y2):
    # Calculate the horizontal and vertical differences
    dx = x2 - x1
    dy = y2 - y1
    # Use the Pythagorean theorem to calculate the distance
    distance = math.sqrt(dx ** 2 + dy ** 2)
    return distance
#@st.cache_data
#@profile
def extract_non_outliers(list):
    # Calculate the quartiles
    q1 = np.percentile(list, 25)
    q3 = np.percentile(list, 75)
    # Calculate the IQR
    iqr = q3 - q1
    # Define the lower and upper bounds for outliers
    lower_bound = q1 - 1 * iqr
    upper_bound = q3 + 1 * iqr
    # Extract values within the middle quartiles and their indices
    filtered_data = []
    filtered_indices = []
    # Filter out the outliers
    for idx, value in enumerate(list):
        if lower_bound <= value <= upper_bound:
            filtered_data.append(value)
            filtered_indices.append(idx)
    return filtered_data, filtered_indices
# @st.cache_data
#@profile
def agglomeration_degree(filtered_labeled_objects, img):
    # 1
    # Apply connected components labeling
    filtered_labeled_objects_u8 = filtered_labeled_objects.astype(np.uint8)
    # Assuming you have contours and the binary image with filled contour regions
    contours, _ = cv2.findContours(filtered_labeled_objects_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # st.write(contours)
    # Function to get neighboring values in an image
    height, width = img.shape
    def get_neighbors(image, x, y, height, width):
        return [image[y + dy, x + dx] for dy in [-1, 0, 1] for dx in [-1, 0, 1] if (dx != 0 or dy != 0) ] # (x != 0 and x!=width and y != 0 and y!=height)and
    # Dictionary to store the number of touching objects and unique values for each contour
    contour_info = {i: {'count': 0, 'unique_values': set()} for i in range(len(contours))}
    # Iterate through each contour
    for i, contour in enumerate(contours):  # Set to store unique non-zero values in the neighborhood
        unique_values_set = set()
        # Iterate through each point in the contour
        for point in contour:
            x, y = point[0]
            print(x,y)
            # Add non-zero values to the set
            neighbors = get_neighbors(filtered_labeled_objects, x, y, height, width)
            unique_values_set.update(val for val in neighbors if val != 0)
        # Store the count and unique values for the contour
        contour_info[i]['count'] = len(unique_values_set)
        contour_info[i]['unique_values'] = unique_values_set
    # Extract counts for plotting
    count_list = [info['count'] for info in contour_info.values()]
    # Use Counter to count occurrences of each element in the list
    occurrences = Counter(count_list)
    # Get unique values and their counts
    unique_values = list(occurrences.keys())
    counts = list(occurrences.values())
    # st.write(unique_values)
    # st.write(counts)
    # Plot histogram
    fig, ax = plt.subplots()
    plt.bar(unique_values, counts)
    # plt.hist(counts, bins=len(set(counts)), align='left', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Touching Objects')
    plt.ylabel('Frequency')
    plt.xticks(unique_values)
    plt.title('Distribution of Agglomerate Sizes')
    st.pyplot(fig)
#---------------------------------------Start Page----------------------------------------------------------#

# with Profiler():
st.set_page_config(layout="wide")
# Sidebar
st.sidebar.header("Image Processing")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "tif"], on_change=st.cache_data.clear() and st.session_state.clear())
# multi scale template matching for finding th size of the scale
template1 = cv2.imread(r'uncontinuous_scale.tiff', cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread(r'continuous_scale.tiff', cv2.IMREAD_GRAYSCALE)
# st.sidebar.write(template1)
# Display the templates on the sidebar
st.sidebar.subheader("Select a Template")
st.sidebar.image(template1, caption='Template 1')
st.sidebar.image(template2, caption='Template 2')
selected_template = st.sidebar.radio("Select a Template", ["Template 1", "Template 2"])
# try:
tab1, tab2 = st.tabs(["\u2001 \u2001\u2001 Analysis \u2001 \u2001 \u2001 ", "Results"])
with tab1:
    st.title("Image Processing App")
    if uploaded_image is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Uploaded Image")
            preprocessed_image = process_image(uploaded_image)
            st.session_state['scale_length'] = find_scale(selected_template, preprocessed_image, template1, template2)
            st.write("Switch values in Size selected Prediction")
            area_diameter = st.toggle('Turn off for Diameter | Turn on for Area')
            # model_change = st.selectbox("Which Model?", ('Basic', 'Fine_Tuned', 'Self_Trained'))
            # st.image(preprocessed_image, use_column_width=True)


        with col2:
            st.subheader("Prediction")
            # Input widgets in the sidebar
            st.sidebar.header("Input Parameters")
            st.session_state['PBS'] = st.sidebar.slider("Probability Score", 0.0, 1.0, 0.3)
            st.session_state['NMS'] = st.sidebar.slider("NMS Score", 0.0, 1.0, 0.2)
            # param3 = st.sidebar.slider("Parameter 3", 0.0, 1.0, 0.5)
            # param4 = st.sidebar.slider("Parameter 4", 0.0, 1.0, 0.5)
            st.session_state['mikro_scale'] = st.sidebar.number_input("Scale in µm", value=5)
            # predicting labels
            # st.write(preprocessed_image)
            # st.write(st.session_state['PBS'], st.session_state['NMS'])
            stardist(preprocessed_image, st.session_state['PBS'], st.session_state['NMS'])
            # labels,details = stardist(preprocessed_image, st.session_state['PBS'], st.session_state['NMS'])
            # st.write("This is Labels:",labels)
            object_sizes, num_objects = display_prediction(st.session_state['scale_length'], st.session_state['mikro_scale'])
        st.subheader("Size Distribution")
        st.markdown("You can adjust the number of bins to change the resolution of the size distribution and the range of object sizes to be displayed.")
        max_bin = int(len(st.session_state['details']['points']))
        max_x = max(object_sizes)
        min_x = min(object_sizes)
        step = int(max_bin / 2.5)
        st.session_state['bins'] = st.slider("Bins", 1, max_bin, step)
        st.session_state['x_range'] = st.slider("Size Range in µm^2",value=[min_x,max_x])
        adjusted_object_sizes = []
        for i, size in enumerate(object_sizes):
            if st.session_state['x_range'][0] <= size <= st.session_state['x_range'][1]:
                adjusted_object_sizes.append((size, i +1))
        plot_hist(adjusted_object_sizes, num_objects)
        with col3:
            st.subheader("Size Selected Prediction")
            x_center_select, y_center_select, x_contour_select, y_contour_select, filtered_labeled_objects = display_selected_labels(adjusted_object_sizes, area_diameter)
    else:
        st.write("Upload an image on the sidebar.")
with tab2:
    st.title("Image Processing App")
    if uploaded_image is not None:
        # st.write(details['coord'].shape)
        # st.write(x_contour_select)
        object_diameters = []
        object_diameters_average = []
        object_diameters_std = []
        # iterating through the centerpoints and contours of the already filtered label data
        for i, (x_center, y_center) in enumerate(zip(x_center_select, y_center_select)):
            diameters = []
            for j, (x_contour, y_contour) in enumerate(zip(x_contour_select[i], y_contour_select[i])):
                diameter = 2*calculate_distance(x_center, y_center, x_contour, y_contour)
                diameters.append(1/(st.session_state['scale_length']/(st.session_state['mikro_scale']*diameter)))
            object_diameters.append(diameters)
            object_diameters_average.append(np.average(diameters))
            object_diameters_std.append(np.std(diameters))
        # get the object_size values from given touple_list
        values_adjusted_object_sizes = [values for values, indices in adjusted_object_sizes]
        values_adjusted_object_sizes,filtered_ind = extract_non_outliers(values_adjusted_object_sizes)
        filtered_std = []
        filtered_object_diameters_average = []
        for i in filtered_ind:
            filtered_object_diameters_average.append(object_diameters_average[i])
            filtered_std.append(object_diameters_std[i])
        column1, column2, column3 = st.columns(3)
        with column1:
            # Display a bar plot
            fig, ax = plt.subplots()
            x = np.arange(len(values_adjusted_object_sizes))
            plt.bar(x, filtered_object_diameters_average, yerr=filtered_std, align='center', alpha=0.5, ecolor='black', capsize=2, error_kw={'elinewidth': 0.8})
            plt.xlabel('Object Nr.')
            plt.ylabel('Diameter in µm')
            plt.title('Average IB Diameter in µm')
            plt.grid()
            st.pyplot(fig)
        with column2:
            # st.write("This is len(adjusted_object_sizes: ", len(values_adjusted_object_sizes), "this is len(filtered_std): ", len(filtered_std))
            # Display a bar plot
            fig, ax = plt.subplots()
            x = np.arange(len(object_diameters_average))
            plt.scatter(values_adjusted_object_sizes, filtered_std)
            plt.xlabel('Object Areas in µm^2')
            plt.ylabel('Standard Deviation of Diameter Distribution in a Object in µm')
            plt.title('Circularity to Area Plot')
            plt.grid()
            st.pyplot(fig)
        with column3:
            fig, ax = plt.subplots()
            x = np.arange(len(filtered_object_diameters_average))
            plt.scatter(filtered_object_diameters_average, filtered_std)
            plt.xlabel('Average Object Diameter in µm')
            plt.ylabel('Standard Deviation of Diameter Distribution in a Object in µm')
            plt.title('Circularity to Average Diameter')
            plt.grid()
            st.pyplot(fig)

        #     agglomeration_degree(filtered_labeled_objects, preprocessed_image)
        overall_average = np.average(object_diameters_average)
        overall_std = np.std(object_diameters_average)
        st.write("Overall Average IB Diameter in µm = ", overall_average)
        st.write("Overall STD = ", overall_std)
    else:
        st.write("Upload an image on the sidebar.")
# except PermissionError as e:
#     st.write("Please reload the page")
#     st.write(e)