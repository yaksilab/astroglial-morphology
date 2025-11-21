import heapq
import math
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage import morphology
from scipy import ndimage
from skimage.measure import regionprops
import logging

logger = logging.getLogger(__name__)


def find_skeleton_endpoints(skeleton):
    """
    Find the endpoints of a skeleton (points with only 1 neighbor).

    Args:
        skeleton: 2D binary array of the skeleton

    Returns:
        list of tuples: [(y, x), (y, x)] - coordinates of the two endpoints
    """
    # Use convolution to find points with exactly 1 neighbor
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)
    endpoints = np.where((skeleton > 0) & (neighbor_count == 1))

    return list(zip(endpoints[0], endpoints[1]))


def measure_local_thickness(mask, point, window_size=20):
    """
    Measure local thickness (compactness) around a point.

    Args:
        mask: 2D binary array of the cell
        point: (y, x) coordinate
        window_size: size of the window around the point

    Returns:
        float: mean distance transform value (thickness measure)
    """
    y, x = point
    h, w = mask.shape

    # Define window bounds
    y_min = max(0, y - window_size // 2)
    y_max = min(h, y + window_size // 2)
    x_min = max(0, x - window_size // 2)
    x_max = min(w, x + window_size // 2)

    # Extract window
    window = mask[y_min:y_max, x_min:x_max]

    if window.sum() == 0:
        return 0

    # Calculate distance transform (measures thickness/distance from boundary)
    dist_transform = ndimage.distance_transform_edt(window)

    # Return mean distance (higher = thicker/more compact)
    return dist_transform[window > 0].mean()  # type: ignore


def _geodesic_distance_on_skeleton(skeleton, start):
    skeleton = skeleton.astype(bool)
    if not skeleton[start]:
        return np.full(skeleton.shape, np.inf, dtype=float)

    h, w = skeleton.shape
    dist = np.full((h, w), np.inf, dtype=float)
    visited = np.zeros((h, w), dtype=bool)
    pq = [(0.0, start[0], start[1])]
    dist[start] = 0.0

    while pq:
        cur_dist, y, x = heapq.heappop(pq)
        if visited[y, x]:
            continue
        visited[y, x] = True

        for ny in range(max(0, y - 1), min(h, y + 2)):
            for nx in range(max(0, x - 1), min(w, x + 2)):
                if not skeleton[ny, nx]:
                    continue
                step = math.hypot(ny - y, nx - x)
                if step == 0:
                    continue
                new_dist = cur_dist + step
                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    heapq.heappush(pq, (new_dist, ny, nx))

    return dist


def measure_thickness_along_skeleton(mask, skeleton, point, distance_from_endpoint=10):
    """
    Measure thickness at a point along the skeleton path, moving inward from endpoint.
    """
    geodesic_dist = _geodesic_distance_on_skeleton(skeleton, point)
    valid_mask = skeleton.astype(bool) & np.isfinite(geodesic_dist)

    if not np.any(valid_mask):
        return 0, point

    ys, xs = np.where(valid_mask)
    distances = geodesic_dist[ys, xs]

    nonzero = distances > 1e-6
    if not np.any(nonzero):
        return 0, point

    ys, xs, distances = ys[nonzero], xs[nonzero], distances[nonzero]
    target_distance = min(distance_from_endpoint, distances.max())
    idx = np.argmin(np.abs(distances - target_distance))
    closest = (int(ys[idx]), int(xs[idx]))

    thickness = measure_local_thickness(mask, closest, window_size=15)
    return thickness, closest


def classify_cells(masks, neck_distance=50):
    """
    Classify cells based on neck thickness.

    Args:
        masks: 2D array where each unique value > 0 is a different cell
        neck_distance: distance in pixels along skeleton from endpoint to measure neck thickness

    Returns:
        dict with classification details
    """
    classifications_pp = {}
    props = regionprops(masks.astype(np.int32))
    classifications = []

    for prop in props:
        cell_label = prop.label
        cell_mask = (masks == cell_label).astype(np.uint8)

        # Skeletonize the cell
        skeleton = morphology.skeletonize(cell_mask)

        # Find endpoints
        endpoints = find_skeleton_endpoints(skeleton)

        if len(endpoints) < 2:
            continue

        # For cells with multiple endpoints, use the two most distant ones
        if len(endpoints) > 2:
            endpoints = sorted(endpoints, key=lambda p: np.sqrt(p[0] ** 2 + p[1] ** 2))
            endpoints = [endpoints[0], endpoints[-1]]

        # Measure neck thickness at specified distance from endpoints
        neck_thickness_1, neck_point_1 = measure_thickness_along_skeleton(
            cell_mask, skeleton, endpoints[0], distance_from_endpoint=neck_distance
        )
        neck_thickness_2, neck_point_2 = measure_thickness_along_skeleton(
            cell_mask, skeleton, endpoints[1], distance_from_endpoint=neck_distance
        )

        # Classify based on neck thickness: thinner neck = soma end
        if neck_thickness_1 < neck_thickness_2:
            soma_end = endpoints[0]
            other_end = endpoints[1]
            soma_neck_point = neck_point_1
            other_neck_point = neck_point_2
            soma_neck_thickness = neck_thickness_1
            other_neck_thickness = neck_thickness_2
            if soma_end[0] < other_end[0]:
                cell_type = "upper"
                classifications.append((1, cell_label))
            else:
                cell_type = "Lower"
                classifications.append((2, cell_label))

        else:
            soma_end = endpoints[1]
            other_end = endpoints[0]
            soma_neck_point = neck_point_2
            other_neck_point = neck_point_1
            soma_neck_thickness = neck_thickness_2
            other_neck_thickness = neck_thickness_1
            if soma_end[0] < other_end[0]:
                cell_type = "upper"
                classifications.append((1, cell_label))
            else:
                cell_type = "lower"
                classifications.append((2, cell_label))

        classifications_pp[cell_label] = {
            "type": cell_type,
            "soma_end": soma_end,
            "process_end": other_end,
            "soma_neck_point": soma_neck_point,
            "other_neck_point": other_neck_point,
            "soma_neck_thickness": soma_neck_thickness,
            "other_neck_thickness": other_neck_thickness,
            "area": prop.area,
            "neck_distance": neck_distance,
        }

    return classifications, classifications_pp


def visualize_classified_cells_with_necks(masks, classifications):
    """
    Visualize cells with soma endpoints, neck measurement points, and skeleton.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))

    # Create colored image for each cell type
    colored_masks = np.zeros((*masks.shape, 3))

    for cell_label, info in classifications.items():
        cell_mask = masks == cell_label

        # Color: Blue for "Type_A", Yellow for "Type_B"
        if info["type"] == "Type_A":
            colored_masks[cell_mask] = [0, 0, 1]  # Blue
        else:
            colored_masks[cell_mask] = [1, 1, 0]  # Yellow

        # Draw skeleton for this cell
        skeleton = morphology.skeletonize(cell_mask.astype(np.uint8))
        skeleton_coords = np.where(skeleton > 0)
        ax.plot(skeleton_coords[1], skeleton_coords[0], "c.", markersize=2, alpha=0.6)

        # Draw soma endpoint (red circle)
        soma_y, soma_x = info["soma_end"]
        ax.plot(
            soma_x,
            soma_y,
            "ro",
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=2,
            label=(
                "Soma endpoint" if cell_label == list(classifications.keys())[0] else ""
            ),
        )

        # Draw process endpoint (green triangle)
        proc_y, proc_x = info["process_end"]
        ax.plot(
            proc_x,
            proc_y,
            "g^",
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=2,
            label=(
                "Process endpoint"
                if cell_label == list(classifications.keys())[0]
                else ""
            ),
        )

        # Draw soma neck measurement point (red square at distance from soma)
        soma_neck_y, soma_neck_x = info["soma_neck_point"]
        ax.plot(
            soma_neck_x,
            soma_neck_y,
            "rs",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
            label=(
                "Soma neck point"
                if cell_label == list(classifications.keys())[0]
                else ""
            ),
        )

        # Draw process neck measurement point (green square at distance from process)
        other_neck_y, other_neck_x = info["other_neck_point"]
        ax.plot(
            other_neck_x,
            other_neck_y,
            "gs",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
            label=(
                "Process neck point"
                if cell_label == list(classifications.keys())[0]
                else ""
            ),
        )

        # Draw lines from endpoints to neck points
        ax.plot(
            [soma_x, soma_neck_x], [soma_y, soma_neck_y], "r-", linewidth=2, alpha=0.6
        )
        ax.plot(
            [proc_x, other_neck_x], [proc_y, other_neck_y], "g-", linewidth=2, alpha=0.6
        )

        # Add text label with classification info
        label_text = f"ID:{cell_label}\n{info['type']}\nSoma neck: {info['soma_neck_thickness']:.2f}px\nNeck dist: {info['neck_distance']}px"
        ax.text(
            soma_x + 20,
            soma_y - 20,
            label_text,
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    ax.imshow(colored_masks)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Cell Classification with Skeleton & Neck Measurements\nBlue=Type_A, Yellow=Type_B\nRed/Green squares=Neck measurement points",
        fontsize=12,
    )
    ax.axis("equal")
    plt.tight_layout()
    plt.show()


# # Test with configurable neck distance
# data_path = r"C:\Users\javid.rezai\YaksiLab\duygu\astroglial-morphology\astroglial-morphology\data\seed4"
# sample = "max_projection_image_seg.npy"

# masks_file = np.load(os.path.join(data_path, sample), allow_pickle=True)
# masks = masks_file.item()["masks"]

# # Classify with neck_distance parameter (adjust as needed)
# neck_distance_pixels = (
#     60  # Change this value to adjust how far from endpoint to measure
# )
# classifications = classify_cells_by_neck_thickness(
#     masks, neck_distance=neck_distance_pixels
# )

# # Print results
# print(f"Total cells found: {len(classifications)}")
# print(f"Neck measurement distance: {neck_distance_pixels} pixels\n")
# print("Classification Results:")
# for cell_label, info in classifications.items():
#     print(f"  Cell {cell_label}: {info['type']}")
#     print(f"    Soma neck thickness: {info['soma_neck_thickness']:.2f}")
#     print(f"    Other neck thickness: {info['other_neck_thickness']:.2f}")
#     print()

# # Visualize
# visualize_classified_cells_with_necks(masks, classifications)
