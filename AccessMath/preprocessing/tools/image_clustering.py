
import math
import cv, cv2
import numpy as np

from sklearn.cluster import AgglomerativeClustering

class ImageClustering:

    @staticmethod
    def cluster_images(images, cluster_features, max_clusters, max_mean_std, debug=False):
        n_points, height, width = images.shape
        aspect_ratio = float(width) / float(height)

        cluster_h = int(math.sqrt(cluster_features / aspect_ratio))
        cluster_w = int(aspect_ratio * cluster_h)

        block_features = np.zeros((n_points, cluster_h * cluster_w), dtype=np.float32)

        for idx in range(n_points):
            gray_scale = cv2.equalizeHist(images[idx, :, :].astype("uint8"))

            small = cv2.resize(gray_scale, (cluster_w, cluster_h))
            block_features[idx, :] = small.reshape(cluster_w * cluster_h)

        n_clusters = 1
        cluster_sizes = None
        cluster_indices = None
        all_valid = False
        while (n_clusters < n_points) and (n_clusters < max_clusters or max_clusters <= 0) and not all_valid:
            n_clusters += 1

            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = clustering.fit_predict(block_features)

            cluster_sizes = [0 for x in range(n_clusters)]
            cluster_indices = [[] for x in range(n_clusters)]

            for idx in range(n_points):
                cluster_sizes[clusters[idx]] += 1
                cluster_indices[clusters[idx]].append(idx)

            all_valid = True
            for cluster_idx in range(n_clusters):
                block_images = np.zeros((cluster_sizes[cluster_idx], height, width), dtype=np.uint8)
                for offset, img_idx in enumerate(cluster_indices[cluster_idx]):
                    img = images[img_idx]
                    block_images[offset, :, :] = img

                cluster_score = np.std(block_images, axis=0).mean()

                if debug:
                    print("Cluster: " + str(cluster_idx) + ", Score: " + str(cluster_score))

                if cluster_score > max_mean_std:
                    all_valid = False
                    break

        if debug:
            # debug, show images in final clusters ...
            for cluster_idx in range(n_clusters):
                for offset, img_idx in enumerate(cluster_indices[cluster_idx]):
                    img = images[img_idx]

                    cv2.imwrite("output/images/x_cluster_" + str(cluster_idx) + "_" + str(img_idx) + ".png", img)

        return cluster_indices