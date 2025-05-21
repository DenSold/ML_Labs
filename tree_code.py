import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    assert len(feature_vector) == len(target_vector)

    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    split_candidates = np.where(sorted_features[:-1] != sorted_features[1:])[0]
    if len(split_candidates) == 0:
        return np.array([]), np.array([]), None, np.inf

    thresholds = (sorted_features[split_candidates] + sorted_features[split_candidates + 1]) / 2
    gini_scores = []

    n_samples = len(target_vector)

    for threshold in thresholds:
        left_indices = sorted_features < threshold
        right_indices = ~left_indices

        if left_indices.sum() == 0 or right_indices.sum() == 0:
            gini_scores.append(np.inf)
            continue

        p_left = np.mean(sorted_targets[left_indices])
        gini_left = 1 - p_left**2 - (1 - p_left)**2

        p_right = np.mean(sorted_targets[right_indices])
        gini_right = 1 - p_right**2 - (1 - p_right)**2

        weighted_gini = (
            (left_indices.sum() / n_samples) * gini_left +
            (right_indices.sum() / n_samples) * gini_right
        )
        gini_scores.append(weighted_gini)

    gini_scores = np.array(gini_scores)
    best_index = np.argmin(gini_scores)

    return thresholds, gini_scores, thresholds[best_index], gini_scores[best_index]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("Неизвестный тип признака")

        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._tree = {}

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = int(sub_y[0])
            return

        if (self._max_depth is not None and depth >= self._max_depth) or len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        best_feature_index, best_threshold, best_gini, best_split_mask = None, None, None, None
        best_cat_mapping = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            else:  # categorical
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {k: clicks.get(k, 0) / counts[k] for k in counts}
                sorted_cats = sorted(ratio, key=ratio.get)
                categories_map = {cat: i for i, cat in enumerate(sorted_cats)}
                feature_vector = np.array([categories_map.get(v, -1) for v in sub_X[:, feature]])

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None or gini is None:
                continue

            if best_gini is None or gini < best_gini:
                best_feature_index = feature
                best_gini = gini
                best_threshold = threshold

                if feature_type == "real":
                    best_split_mask = feature_vector < threshold
                    best_cat_mapping = None
                else:
                    best_split_mask = feature_vector < threshold
                    best_cat_mapping = categories_map

        if best_feature_index is None or best_split_mask.sum() < self._min_samples_leaf or (~best_split_mask).sum() < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        node["type"] = "nonterminal"
        node["feature_split"] = best_feature_index
        feature_type = self._feature_types[best_feature_index]

        if feature_type == "real":
            node["threshold"] = float(best_threshold)
        else:
            node["categories_split"] = [k for k, v in best_cat_mapping.items() if v < best_threshold]

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[best_split_mask], sub_y[best_split_mask], node["left_child"], depth + 1)
        self._fit_node(sub_X[~best_split_mask], sub_y[~best_split_mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_index = node["feature_split"]
        feature_type = self._feature_types[feature_index]

        if feature_type == "real":
            threshold = node.get("threshold")
            return self._predict_node(
                x, node["left_child"] if float(x[feature_index]) < threshold else node["right_child"]
            )
        else:
            category = x[feature_index]
            left_categories = node.get("categories_split", [])
            return self._predict_node(
                x, node["left_child"] if category in left_categories else node["right_child"]
            )

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X], dtype=int)
