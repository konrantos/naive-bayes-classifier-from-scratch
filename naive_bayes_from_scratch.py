# Naive Bayes classifier demo on the Iris dataset
# ---------------------------------------------------
# The script reads the Iris CSV file (with a custom header format),
# splits it into training and test sets, learns probability
# parameters for discrete and continuous features, evaluates the
# test accuracy, and then enters an interactive loop so you can
# classify new samples from the keyboard.

import csv
import math
import random

# ---------------------------------------------------
# 1. Data loading utilities
# ---------------------------------------------------
def load_dataset(filename):
    """Load a dataset whose first row alternates *feature name*, *type*
    ("C" for continuous, "D" for discrete) and ends with the class name."""
    features_list = []
    data_list = []

    # Open the CSV file and read the header row
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        first_row = next(reader)

        # Build metadata for feature columns
        for i in range(0, len(first_row) - 1, 2):
            features_list.append({"name": first_row[i], "type": first_row[i + 1]})

        # The last column is always the class label
        features_list.append({"name": first_row[-1], "type": "class"})

        # Read the remaining data rows
        for row in reader:
            data_list.append(row)

    return features_list, data_list


# Actually load the Iris data -------------------------------------------------
features_list, data_list = load_dataset("iris.csv")

print("Features:")
for f in features_list:
    print(f)

print("\nData samples:")
for row in data_list[:5]:
    print(row)

# ---------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------
# Make the split reproducible
random.seed(0)
# Shuffle the dataset in-place
random.shuffle(data_list)
# Use 70 % of the data for training
split = int(0.7 * len(data_list))
training_set = data_list[:split]
# The rest goes into the test set
test_set = data_list[split:]

print("\nTraining set size:", len(training_set))
print("Test set size:", len(test_set))

# ---------------------------------------------------
# 3. Inspect feature domains (needed later for PDFs)
# ---------------------------------------------------
def feature_info(features_list, data_list):
    """Return augmented metadata with min/max or value lists."""
    features_info_list = []
    for i, feature in enumerate(features_list):
        new_feat = {"name": feature["name"], "type": feature["type"]}

        # Handle categorical (discrete) features and the class label
        if feature["type"] in ("D", "class"):
            new_feat["values"] = list({row[i] for row in data_list})
        # Handle continuous features
        elif feature["type"] == "C":
            numbers = [float(row[i]) for row in data_list]
            new_feat["min"] = min(numbers)
            new_feat["max"] = max(numbers)
        features_info_list.append(new_feat)

    return features_info_list


features_info_list = feature_info(features_list, training_set)

print("\nEnriched feature list:")
for f in features_info_list:
    print(f)

# ---------------------------------------------------
# 4. Estimate probabilities required for Naive Bayes
# ---------------------------------------------------
def a_priori(features_info_list, data_list):
    """Compute P(class) for every class value."""
    out = []
    class_index = len(features_info_list) - 1
    class_values = features_info_list[class_index]["values"]
    total = len(data_list)

    # Raw counts per class
    counts = {cl: 0 for cl in class_values}
    for row in data_list:
        counts[row[class_index]] += 1

    # Convert counts to probabilities
    for cl in class_values:
        out.append({"class": cl, "probability": counts[cl] / total})
    return out


a_priori_prob_list = a_priori(features_info_list, training_set)

print("\nA priori class probabilities:")
for p in a_priori_prob_list:
    print(f"{p['class']}: {p['probability']:.3f}")

# ---------------------------------------------------
# 4a. Discrete-feature conditional probabilities
#     (with Laplace smoothing where needed)
# ---------------------------------------------------
def discrete_with_smoothing(features_info_list, data_list):
    """Return counts and P(x|class) for every discrete feature value."""
    discrete_count_list = []
    discrete_probs_list = []

    class_index = len(features_info_list) - 1
    class_values = {row[class_index] for row in data_list}

    # Pre-compute class counts for denominators
    class_counts = {c: 0 for c in class_values}
    for row in data_list:
        class_counts[row[class_index]] += 1

    for i, feature in enumerate(features_info_list[:-1]):
        # Skip continuous features here
        if feature["type"] != "D":
            continue

        # Build a 2-D count table (value × class)
        counts = {}
        has_zero = False
        for row in data_list:
            val = row[i]
            cls = row[class_index]
            counts.setdefault(val, {c: 0 for c in class_values})
            counts[val][cls] += 1
            if counts[val][cls] == 0:
                has_zero = True

        # Save raw counts (handy for debugging)
        for val in counts:
            for cls in class_values:
                discrete_count_list.append(
                    {
                        "feature": feature["name"],
                        "value": val,
                        "class": cls,
                        "count": counts[val][cls],
                    }
                )

        # Convert to probabilities (additive smoothing if any 0 appears)
        k = len(counts)  # Number of possible values
        p = 1 / k        # Prior probability under a uniform assumption
        a = 3            # Strength of the prior (hyper-parameter)
        for val in counts:
            for cls in class_values:
                if has_zero:
                    prob = (counts[val][cls] + a * p) / (class_counts[cls] + a)
                else:
                    prob = counts[val][cls] / class_counts[cls]

                discrete_probs_list.append(
                    {
                        "feature": feature["name"],
                        "value": val,
                        "class": cls,
                        "probability": prob,
                    }
                )

    return discrete_count_list, discrete_probs_list


discrete_count_list, discrete_probs_list = discrete_with_smoothing(
    features_info_list, training_set
)

print("\nCounts (before smoothing):")
for c in discrete_count_list:
    print(f"{c['feature']} = {c['value']} | {c['class']} → {c['count']}")

print("\nP(x|class) for discrete features:")
for p in discrete_probs_list:
    print(f"{p['feature']} = {p['value']} | {p['class']} → {p['probability']:.3f}")

# ---------------------------------------------------
# 4b. Continuous-feature conditional densities (Gaussian)
# ---------------------------------------------------
def continuous_with_gaussian(features_info_list, data_list):
    """Fit mean/variance per class for every continuous feature."""
    gaussian_params_list = []
    class_index = len(features_info_list) - 1
    class_values = {row[class_index] for row in data_list}

    for i, feature in enumerate(features_info_list[:-1]):
        # Skip discrete features
        if feature["type"] != "C":
            continue

        # Loop over every class value
        for cls in class_values:
            values = [float(row[i]) for row in data_list if row[class_index] == cls]
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            gaussian_params_list.append(
                {
                    "feature": feature["name"],
                    "class": cls,
                    "mean": mean,
                    "variance": variance,
                }
            )

    return gaussian_params_list


gaussian_params_list = continuous_with_gaussian(features_info_list, training_set)

print("\nGaussian parameters for continuous features:")
for g in gaussian_params_list:
    print(
        f"{g['feature']} | {g['class']} → mean={g['mean']:.3f}, var={g['variance']:.3f}"
    )

# ---------------------------------------------------
# 5. Model evaluation on the hold-out test set
# ---------------------------------------------------
def evaluate(
    test_set,
    features_info_list,
    discrete_probs,
    gaussian_params,
    a_priori_prob_list,
):
    """Classify every sample in *test_set* and report accuracy."""
    correct = 0
    total = len(test_set)
    class_values = [item["class"] for item in a_priori_prob_list]
    errors = []  # Collect mis-classified samples for inspection

    for row in test_set:
        sample = row[:-1]
        true_class = row[-1]

        # Start with P(class) for each class value
        class_scores = {
            c: next(item["probability"] for item in a_priori_prob_list if item["class"] == c)
            for c in class_values
        }

        # Multiply by P(x|class) for every feature (naive independence)
        for c in class_values:
            prob = class_scores[c]
            for i, feature in enumerate(features_info_list[:-1]):
                name = feature["name"]
                ftype = feature["type"]
                value = sample[i]

                if ftype == "D":  # discrete feature
                    match = next(
                        item
                        for item in discrete_probs
                        if item["feature"] == name
                        and item["value"] == value
                        and item["class"] == c
                    )
                    prob *= match["probability"]
                else:  # continuous feature – use Gaussian PDF
                    match = next(
                        item
                        for item in gaussian_params
                        if item["feature"] == name and item["class"] == c
                    )
                    mean, var = match["mean"], match["variance"]
                    x = float(value)
                    gauss = (
                        1 / math.sqrt(2 * math.pi * var)
                    ) * math.exp(-((x - mean) ** 2) / (2 * var))
                    prob *= gauss

            class_scores[c] = prob

        # Choose the arg-max
        predicted = max(class_scores, key=class_scores.get)
        if predicted == true_class:
            correct += 1
        else:
            errors.append((sample, true_class, predicted))

    # Show mis-classifications (usually none for Iris)
    if errors:
        print("\nMisclassified samples:")
        for sample, actual, pred in errors:
            print(f"  Sample={sample}  True={actual}  Predicted={pred}")

    accuracy = correct / total
    print(f"\nTest set accuracy: {accuracy:.2%}")


# Evaluate once ---------------------------------------------------------------
evaluate(
    test_set,
    features_info_list,
    discrete_probs_list,
    gaussian_params_list,
    a_priori_prob_list,
)

# ---------------------------------------------------
# 6. Interactive prediction loop
# ---------------------------------------------------
def predict_class(
    features_info_list, discrete_probs, gaussian_params, a_priori_prob_list
):
    """Prompt the user for feature values and output posterior percentages."""
    class_scores = {}
    class_values = [item["class"] for item in a_priori_prob_list]
    user_input = []

    # Gather inputs feature-by-feature -----------------------------
    for i, feature in enumerate(features_info_list[:-1]):
        name, ftype = feature["name"], feature["type"]
        if ftype == "C":
            extra = f"(min: {feature['min']}, max: {feature['max']})"
        else:  # discrete choices
            extra = f"({', '.join(feature['values'])})"

        while True:
            value = input(f"Enter value for {name} {extra}: ").strip()
            if ftype == "C":
                try:
                    num = float(value)
                    if feature["min"] <= num <= feature["max"]:
                        user_input.append(num)
                        break
                except ValueError:
                    pass
                print("Invalid numeric input. Try again.")
            else:  # Discrete
                if value in feature["values"]:
                    user_input.append(value)
                    break
                print("Invalid option. Try again.")

    # Compute posterior un-normalised scores ----------------------
    for c in class_values:
        prob = next(
            item["probability"] for item in a_priori_prob_list if item["class"] == c
        )
        for i, feature in enumerate(features_info_list[:-1]):
            name, ftype = feature["name"], feature["type"]
            value = user_input[i]
            if ftype == "D":
                match = next(
                    item
                    for item in discrete_probs
                    if item["feature"] == name
                    and item["value"] == value
                    and item["class"] == c
                )
                prob *= match["probability"]
            else:  # Gaussian PDF
                match = next(
                    item
                    for item in gaussian_params
                    if item["feature"] == name and item["class"] == c
                )
                mean, var = match["mean"], match["variance"]
                x = value
                gauss = (
                    1 / math.sqrt(2 * math.pi * var)
                ) * math.exp(-((x - mean) ** 2) / (2 * var))
                prob *= gauss
        class_scores[c] = prob

    # Normalise to percentages ------------------------------------
    total_prob = sum(class_scores.values())
    for c in class_scores:
        class_scores[c] = round(100 * class_scores[c] / total_prob, 2)

    print("\nPosterior probabilities:")
    for c in class_values:
        print(f"{c}: {class_scores[c]}%")


# Main interactive loop -------------------------------------------------------
while True:
    print("\nEnter values for a new instance:")
    predict_class(
        features_info_list,
        discrete_probs_list,
        gaussian_params_list,
        a_priori_prob_list,
    )
    while True:
        answer = input("\nDo you want to classify another sample? (y/n): ").strip().lower()
        if answer in {"y", "n"}:
            break
        print("Invalid response. Please enter 'y' or 'n'.")
    if answer == "n":
        break
