import cv2 # OpenCV for image processing
import numpy as np # NumPy for numerical operations
import re # Regular expressions for string matching
import os # OS for file operations
import svgwrite # SVG writing library
from matplotlib import pyplot as plt # Matplotlib for plotting
from svg.path import parse_path, Move, Line, CubicBezier, QuadraticBezier, Arc # SVG path parsing
from scipy.signal import argrelextrema # Local extrema detection
from scipy.ndimage import gaussian_filter1d # Gaussian filter for smoothing
from glob import glob # Glob for file pattern matching
from xml.dom import minidom # XML parsing for SVG files

# === CONFIGURATION ===
KNOWN_WIDTH_CM = 21.0           # A4 width in cm
FALLBACK_PX_TO_CM = 0.0264      # Fallback scale
MIN_CONTOUR_AREA = 500         # Minimum area to consider
ARM_REJECT_ASPECT = 1.5       # Wide shapes (likely arms) filtered out

# Global scale
PIXEL_TO_CM = FALLBACK_PX_TO_CM
# Set the pixel to cm scale based on A4 paper width

# Load SVG contours from a file
def load_svg_contours(svg_path):
    # Check if SVG file exists
    contours = []
    # Parse the SVG file
    doc = minidom.parse(svg_path)
    
    # Extract polygons (straightforward)
    for poly in doc.getElementsByTagName('polygon'):
        # Check if polygon has points
        pts = poly.getAttribute('points').strip().split()
        # Skip if no points found
        coords = np.array([[float(x), float(y)] for x,y in (p.split(',') for p in pts)], dtype=np.float32)
        # Skip if not enough points
        if len(coords) >= 3:
            # Reshape for OpenCV: (num_points,1,2)
            contours.append(coords.reshape(-1, 1, 2))
    
    # Extract paths, split into subpaths by 'Move' commands
    for path_tag in doc.getElementsByTagName('path'):
        # Check if path has 'd' attribute
        d = path_tag.getAttribute('d')
        # Skip if no 'd' attribute
        if not d:
            continue
        # Parse the path data
        path = parse_path(d)
        
        # Collect points for current subpath
        subpath_points = []
        # Initialize list to hold all subpaths
        all_subpaths = []
        # Iterate through path segments
        for seg in path:
            # Check if segment is a Move command
            if isinstance(seg, Move):
                # If current subpath has points, save it and start new
                if subpath_points:
                    all_subpaths.append(subpath_points)
                    # Reset subpath points
                    subpath_points = []
                # Start new subpath with Move point
                subpath_points.append([seg.end.real, seg.end.imag])
            else:
                # Sample points along this segment
                samples = 30
                # Handle different segment types
                for t in np.linspace(0, 1, samples):
                    # Get point on segment
                    pt = seg.point(t)
                    # Append point to current subpath
                    subpath_points.append([pt.real, pt.imag])
        
        # Add last subpath
        if subpath_points:
            all_subpaths.append(subpath_points)
        
        # Convert all subpaths to contours, ensure closed polygons
        for pts in all_subpaths:
            pts = np.array(pts, dtype=np.float32)
            # Close polygon if not already
            if not np.allclose(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[0]])
            # Reshape for OpenCV: (num_points,1,2)
            if len(pts) >= 3:
                contours.append(pts.reshape(-1, 1, 2))
    # Check if any contours were found
    doc.unlink()
    # Return the list of contours
    return contours

# Normalize contour to center it around (0,0) and scale to unit length
def center_contour(cnt):
    # Check if contour is empty
    cnt = np.asarray (cnt).reshape(-1, 2).astype(np.float32)
    # Check if contour is empty
    if cnt.shape[0]<3:
        return None
    # Calculate centroid
    M = cv2.moments(cnt)
    # Check if contour is empty
    if M["m00"] == 0:
        return None
    # Calculate centroid coordinates
    cx = M["m10"]/M["m00"]
    cy = M["m01"]/M["m00"]
    # Center the contour around (0,0)
    centered = cnt - np.array([[cx,cy]], dtype=np.float32)
    # Calculate the maximum distance from the origin
    return centered

# Detect whether a contour's neck is V-shaped or round-shaped
def detect_neck_type(contour, plot=False, angle_threshold=60):
    """
    Detect whether a contour's neck is V-shaped or-round-shaped.
    """
    # Reshape contour if needed
    if contour.ndim == 3:
        pts = contour.reshape(-1, 2)
    else:
        pts = contour.copy()

    # Sort points by x
    pts_sorted = pts[np.argsort(pts[:, 0])]

    # Filter for upper-middle section (say upper 25%)
    min_y = np.min(pts_sorted[:, 1])
    max_y = np.max(pts_sorted[:, 1])
    # Define the upper quarter of the y-range
    quarter = min_y + 0.25 * (max_y - min_y)
    # Select points in the upper section
    upper_section = pts_sorted[pts_sorted[:, 1] < quarter]

    # Find minimum (valley)
    min_idx = np.argmin(upper_section[:, 1])
    # Check if we have enough points to determine left/right neighbors
    valley = upper_section[min_idx]
    # Check if we have enough points to determine left/right neighbors
    if len(upper_section) < 5:
        print("Not enough points to determine.")
        return "Unknown"
    # Get distances in x-direction from valley
    x_dists = np.abs(upper_section[:, 0] - valley[0])

    # Get indices of 2 closest points (excluding the valley itself)
    sorted_indices = np.argsort(x_dists)
    neighbors = upper_section[sorted_indices[1:3]]  # skip index 0 (valley itself)

    # Assign left/right based on x
    if neighbors[0][0] < valley[0]:
        # Ensure left is always the one with smaller x
        left = neighbors[0]
        right = neighbors[1]
    else:
        # Ensure left is always the one with smaller x
        left = neighbors[1]
        right = neighbors[0]

    # Calculate angle
    vector1 = left - valley
    vector2 = right - valley
    # Check if vectors are valid
    dot = np.dot(vector1, vector2)
    # Calculate magnitudes of vectors
    mag1 = np.linalg.norm(vector1)
    mag2 = np.linalg.norm(vector2)
    # Handle zero magnitude to avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return "Unknown"
    # Clip to avoid numerical issues with arccos
    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    # Calculate angle in degrees
    angle = np.degrees(np.arccos(cos_angle))
    # Print angle for debugging
    print(f"Valley angle = {angle}")
    # Plot if requested
    if angle < angle_threshold:
        return "Vneck"
    return "Round"

# Detect pocket type based on contour shape and aspect ratio
def detect_pocket_type(contour, img_shape):
    # Check if contour is empty
    h,w = img_shape[:2]
    # Create a mask for the contour
    mask = np.zeros((h,w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # Focus on chest region
    cy1, cy2 = int(h*0.25), int(h*0.5)
    cx1, cx2 = int(w*0.3), int(w*0.7)
    # Extract region of interest (ROI) for pocket detection
    reg = mask[cy1:cy2, cx1:cx2]
    # Check if region is empty
    cnts, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check if contours are empty
    pockets=[]
    # Iterate through contours to classify pocket types
    for c in cnts:
        ar = cv2.contourArea(c)
        # Skip small contours
        if ar<150: continue
        # Calculate centroid
        x,y,bw,bh = cv2.boundingRect(c)
        # Calculate centroid coordinates
        asp = bh/(bw+1e-5)
        # Skip contours that are too wide (likely arms)
        c += [cx1, cy1]
        # Check aspect ratio to classify pocket type
        if 0.8<=asp<=1.3: pockets.append(("Square",c))
        elif asp>1.3: pockets.append(("Angle",c))
    # Check if any pockets were found
    if not pockets: return "NoPocket"
    # Sort pockets by area in descending order
    pockets.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)
    # Return the largest pocket type
    return pockets[0][0]

# Detect pocket presence using color and edge detection
def detect_pocket_with_color(image, contour):
    # Check if contour is empty
    h,w = image.shape[:2]
    # Create a mask for the contour
    mask = np.zero((h,w), np.uint8)
    # Check if contour is empty
    cv2.drawContours(mask, [contour], -1, 255, -1)

    #focus on chest region
    cy1,cy2 = int(h*0.25), int(h*0.5)
    cx1, cx2 = int(w*0.3), int(w*0.7)
    roi = image[cy1:cy2 , cx1:cx2]
    roi_mask = mask[cy1:cy2, cx1:cx2]

    #convert to grayscale & apply mask
    gray = cv2.cvtColor(roi, cv2.COLOR_BG2BGR)
    masked_gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

    #Detect strong edges
    edges = cv2.Canny(masked_gray, 50, 150)

    #Now find pocket like contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area
    for c in cnts:
        # Check if contour is empty
        area = cv2.contourArea(c)
        # Skip small contours
        if 150<area<2000:
            return "Pocket"
        return "NoPocket"
    
# Calculate area in cm² based on pixel count
def calculate_area_cm2(contour):
    # Check if contour is empty
    return cv2.contourArea(contour) * PIXEL_TO_CM**2


# Find the reference scale based on A4 paper width
def find_reference_scale(image, contours):
    # Check if contours are empty
    global PIXEL_TO_CM
    # Check if contours are empty
    for c in contours:
        # Skip small contours
        _,_,w,h = cv2.boundingRect(c)
        # Skip contours that are too small in bounding box
        ar = w/float(h)
        # Skip contours that are too wide (likely arms)
        area = cv2.contourArea(c)
        # Check if aspect ratio and area match A4 paper
        if 0.65<ar<0.75 and area>5000:
            # Calculate the pixel to cm scale based on A4 width
            PIXEL_TO_CM = KNOWN_WIDTH_CM / w
            # Print the scale factor
            print(f"[SCALE] A4 width {w}px → {PIXEL_TO_CM:.2f}px/cm")
            # Return early since we found the A4 contour
            return
    # If no A4 contour found, use fallback scale
    print("[SCALE] A4 not detected; using fallback.")

# Save contours to SVG file
def save_svg(contours, fn, shape):
    # Check if contours are empty
    h,w = shape[:2]
    # If no contours, create an empty SVG
    dwg = svgwrite.Drawing(fn, size=(w,h))
    # Add a background rectangle to ensure transparency
    for c in contours:
        # Check if contour is empty
        pts = [(int(p[0][0]), int(p[0][1])) for p in c]
        # Skip empty contours
        dwg.add(dwg.polygon(pts, fill='none', stroke='black', stroke_width=1))
        # Add a polygon for each contour
    dwg.save()
    # Print confirmation
    print(f"[SVG] Saved: {fn}")

# Match garment contours to SVG templates
def match_to_templates(image_path, svg_folder):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    # Check if image is loaded
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE to enhance contrast in seam areas
    blur = cv2.GaussianBlur(gray, (3,5),0)
    #reduce noise while preserving edges
    edges = cv2.Canny(blur,25,120)
    # Find contours in the edge-detected image
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area
    cnts = [c for c in cnts if cv2.contourArea(c)>100]
    # Check if any contours were found
    if not cnts:
        print("No contours in garment detection."); return
    # Find the largest contour, assuming it's the garment
    garment = max(cnts, key=cv2.contourArea)
    #create transparent
    h,w = img.shape[:2]
    # Create a transparent output image
    output = np.zeros((h, w, 4), dtype = np.uint8)
    # Fill the output with the original image
    for c in cnts:
        cv2.drawContours(output, [c], -1, (0, 0, 0, 255), thickness=1)
    # Draw the garment contour in white
    seam_edges = cv2.Canny(gray, 25, 120)
    # Draw the seam edges on the output image
    seam_coords = np.column_stack(np.where(seam_edges> 0))
    # Draw the seam line in white
    for y, x in seam_coords:
        output[y, x] = [255, 255, 255, 255]
    # Draw the garment contour in white
    cv2.imwrite("output_with_seam.jpg", output)
    # Save the output image with transparent background
    print ("[info]save image with transparent backgorund and seam line.")

    # Detect garment features
    neck = detect_neck_type(garment)
    # Detect pocket type
    pk = detect_pocket_type(garment, img.shape)
    # If no pocket type detected, try color detection
    if pk == "No pocket":
        pk = detect_pocket_with_color(img, garment)
    # Calculate area in cm²
    area_cm2 = calculate_area_cm2(garment)
    print(f"Neck:{neck}, Pocket:{pk}, Size:{area_cm2:.1f}cm²")

    # Construct case-sensitive filename prefix
    if pk == "NoPocket":
        name_base = f"NECKL_{neck}_59"
    else:
        name_base = f"NECKL_{neck}_POCKET_{pk}_59"

    print(f"Looking for SVG files starting with: {name_base}")

    # Case-sensitive match
    cands = [f for f in os.listdir(svg_folder) if f.startswith(name_base) and f.endswith(".svg")]

    # Fallback to all SVGs if no match
    if not cands:
        print("No name-match; trying all.")
        cands = [f for f in os.listdir(svg_folder) if f.endswith(".svg")]

    # Sort by _59_1, _59_2, ...
    def extract_number(filename):
        # Extract the number after _59_ using regex
        match = re.search(r'_59_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')
    # Sort candidates by the number after _59_
    cands.sort(key=extract_number)

    # Proceed with best match
    if cands:
        # Use the first candidate as the best match
        best_match_file = os.path.join(svg_folder, cands[0])
        # Load contours from the best match SVG
        target_contour = garment
        # Normalize the target contour
        target_area = calculate_area_cm2(target_contour)
        # Normalize the target contour to center it
        tnorm = center_contour(target_contour)
        # Check if normalization was successful
        best_match_file = None
        # Initialize best score
        best_score = float('inf')
        # Initialize best area difference and shape score
        for svg_file in cands:
            # Load contours from the SVG file
            svg_path = os.path.join(svg_folder, svg_file)
            # Load contours from the SVG file
            svg_contours = load_svg_contours(svg_path)
            # Check if any contours were loaded
            if not svg_contours:
                continue
            # Normalize each SVG contour and calculate scores
            for sc in svg_contours:
                # Skip small contours
                if cv2.contourArea(sc) < 100:
                    continue
                # Normalize the SVG contour
                sc_norm = center_contour(sc)
                # Skip if normalization failed
                if sc_norm is None:
                    continue
                # Calculate area in cm²
                sc_area = calculate_area_cm2(sc)
                # Calculate area difference
                area_diff = abs(target_area - sc_area)

                # Shape similarity: lower is better
                shape_score = cv2.matchShapes(tnorm, sc_norm, cv2.CONTOURS_MATCH_I1, 0)

                # Combine both into a weighted score
                total_score = shape_score * 10 + area_diff  # adjust weights if needed
                # Check if this is the best match so far
                if total_score < best_score:
                    best_score = total_score
                    best_match_file = svg_path
                    best_area_diff = area_diff
                    best_shape_score = shape_score
        # Print the best match details
        if best_match_file:
            print(f"[MATCHED] Best match: {best_match_file}")
            print(f"         Area diff = {best_area_diff:.2f} cm², Shape score = {best_shape_score:.4f}, Combined = {best_score:.2f}")
        else:
            print("No valid SVG match found.")
    else:
        print("No SVG templates found.")
        
# Process the image to extract parts and save as SVG
def process_image(image_path, output_svg_dir, template_folder):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    # Check if image is loaded
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Apply clahe to enhance contrast in seam areas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # Apply CLAHE to the grayscale image
    gray = clahe.apply(gray)
    #reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Apply Gaussian blur to reduce noise
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # Apply Canny edge detection to find contours
    edges = cv2.Canny(blur, 50, 150)
    # Find contours in the edge-detected image
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area
    find_reference_scale(img, cnts)
    parts = []
    # Sort contours by area
    vis = img.copy()
    # Filter and process contours
    for i,c in enumerate(cnts):
        # Skip small contours
        area = cv2.contourArea(c)
        # Skip contours that are too small
        if area < MIN_CONTOUR_AREA:
            continue
        # Skip contours that are too large
        eps = 0.001*cv2.arcLength(c, True)
        # Approximate the contour to reduce points
        approx = cv2.approxPolyDP(c, eps, True)
        # Skip contours with too few points
        x,y,w,h = cv2.boundingRect(approx)
        # Skip contours that are too small in bounding box
        if w/float(h) > ARM_REJECT_ASPECT:
            continue  # Likely arms
        parts.append((i, approx, (x,y,w,h)))
    # Sort parts by bounding box area
    os.makedirs(output_svg_dir, exist_ok=True)
    extracted = []
    for idx, cnt, (x,y,w,h) in parts:
        # Skip contours that are too small in bounding box
        mask = np.zeros(img.shape[:2], np.uint8)
        # Create a mask for the contour
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        # Extract the sub-image using the mask
        sub = cv2.bitwise_and(img, img, mask=mask)[y:y+h, x:x+w]
        # Skip if the sub-image is too small
        gray_s = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        _,bin_s = cv2.threshold(gray_s,128,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        # Find contours in the sub-image
        cnts2,_ = cv2.findContours(bin_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Skip if no contours found in sub-image
        svg_fn = os.path.join(output_svg_dir, f"part_{idx+1}.svg")
        # Save the sub-image as SVG
        save_svg(cnts2, svg_fn, sub.shape)
        # Draw the contour on the original image for visualization
        cv2.drawContours(vis, [cnt], -1, (0,255,0), 2)
        # Draw bounding box on the original image
        cv2.putText(vis, f"P{idx+1}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        # Append the extracted part
        extracted.append((svg_fn, sub))

    # Display results
    cols = 3
    # Calculate number of rows needed for subplots
    rows = (len(extracted)+cols-1)//cols
    # Create a figure to display the results
    plt.figure(figsize=(15,5 + rows*2))
    # Show the original image with detected parts
    plt.subplot(rows+1, cols, 1)
    # Visualize the original image with detected parts
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    # Set title and hide axes
    plt.title("Detected Parts")
    # Hide axes for the original image
    plt.axis('off')
    # Show each extracted part
    for j, (svg_fn, imgp) in enumerate(extracted, start=2):
        # Show each extracted part in a subplot
        plt.subplot(rows+1, cols, j)
        # Convert BGR to RGB for displaying
        plt.imshow(cv2.cvtColor(imgp, cv2.COLOR_BGR2GRAY), cmap='gray')
        # Set title to the SVG filename
        plt.title(os.path.basename(svg_fn))
        # Hide axes for the extracted part
        plt.axis('off')
    # Show the last subplot with the original image
    plt.tight_layout()
    # Show the plot
    plt.show()
    # Match the garment contours to SVG templates
    match_to_templates(image_path, template_folder)

if __name__ == "__main__":
    # image loading and processing
    IMAGE = r"C:\Users\HsuMyatWin\seam\GreenTshirt_58cm.jpg"
    # Define output directory for SVG files
    OUTPUT_SVG = r"C:\Users\HsuMyatWin\seam\output_svgs"
    # Define the folder containing SVG templates
    TEMPLATES = r"C:\Users\HsuMyatWin\OneDrive - Fitdex (DIVI Grupa)\Andrejs Neimanis's files - Work script folder\Hsu_Myat_Win\TShirts\svgFiles"
    # Process the image and save results
    process_image(IMAGE, OUTPUT_SVG, TEMPLATES)

