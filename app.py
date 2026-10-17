import streamlit as st
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Smart Animal Image Search",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 Smart Animal Image Search")
st.write(
    "Upload an animal image to find visually similar "
    "images from the Animals-10 dataset."
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset"

MAX_IMAGES = 500
TOP_RESULTS = 5

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

FOLDER_CATEGORIES = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

CATEGORY_NAMES = {
    "dog": "Dogs",
    "cat": "Cats",
    "horse": "Horses",
    "elephant": "Elephants",
    "butterfly": "Butterflies",
    "chicken": "Chickens",
    "cow": "Cows",
    "sheep": "Sheep",
    "spider": "Spiders",
    "squirrel": "Squirrels"
}

@st.cache_resource
def load_models():
    weights = ResNet18_Weights.DEFAULT

    classifier = resnet18(weights=weights)
    classifier.eval()

    feature_model = nn.Sequential(
        *list(classifier.children())[:-1]
    )
    feature_model.eval()

    preprocess = weights.transforms()
    labels = weights.meta["categories"]

    return classifier, feature_model, preprocess, labels

def identify_category(label):
    name = label.lower()

    cat_words = [
        "tabby", "tiger cat", "persian cat",
        "siamese cat", "egyptian cat", "lynx",
        "cougar", "lion", "tiger", "leopard",
        "snow leopard", "jaguar", "cheetah"
    ]

    if any(word in name for word in cat_words):
        return "cat"

    dog_words = [
        "dog", "retriever", "terrier", "spaniel",
        "shepherd", "collie", "poodle", "pug",
        "beagle", "bulldog", "chihuahua", "husky",
        "malamute", "rottweiler", "doberman",
        "pinscher", "schnauzer", "greyhound",
        "whippet", "boxer", "mastiff", "dachshund",
        "corgi", "dalmatian", "papillon", "pekinese",
        "shih-tzu", "chow", "akita", "samoyed",
        "keeshond", "briard", "basenji",
        "affenpinscher", "mexican hairless",
        "wolfhound", "tervuren", "schipperke",
        "lhasa", "vizsla", "kuvasz", "malinois"
    ]

    if any(word in name for word in dog_words):
        return "dog"

    if any(word in name for word in [
        "horse", "sorrel", "stallion", "colt", "mare"
    ]):
        return "horse"

    if "elephant" in name:
        return "elephant"

    if any(word in name for word in [
        "butterfly", "monarch", "sulphur butterfly",
        "lycaenid"
    ]):
        return "butterfly"

    if any(word in name for word in [
        "hen", "cock", "rooster", "chicken"
    ]):
        return "chicken"

    if any(word in name for word in [
        "cow", "ox", "bull", "bison", "water buffalo"
    ]):
        return "cow"

    if any(word in name for word in [
        "sheep", "ram", "bighorn", "wether"
    ]):
        return "sheep"

    if any(word in name for word in [
        "spider", "tarantula", "garden spider",
        "wolf spider", "black widow"
    ]):
        return "spider"

    if any(word in name for word in [
        "squirrel", "marmot", "chipmunk"
    ]):
        return "squirrel"

    return None


def predict_animal(image, classifier, preprocess, labels):
    tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        output = classifier(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    top_indices = torch.argsort(
        probabilities, descending=True
    )[:10].tolist()

    for index in top_indices:
        category = identify_category(labels[index])

        if category is not None:
            return category, probabilities[index].item(), labels[index]

    return None, 0.0, labels[top_indices[0]]

def extract_features(image, feature_model, preprocess):
    tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        features = feature_model(tensor)

    vector = features.flatten().cpu().numpy()

    return vector / (np.linalg.norm(vector) + 1e-10)

@st.cache_data(show_spinner=False)
def get_image_paths(dataset_path, max_images):
    root = Path(dataset_path)

    if not root.exists():
        return []

    paths = []
    categories = list(FOLDER_CATEGORIES.keys())

    images_per_category = max(
        1, max_images // len(categories)
    )

    for folder_name in categories:
        folder = root / folder_name

        if not folder.exists():
            continue

        category_images = sorted(
            path for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in EXTENSIONS
        )

        selected = category_images[:images_per_category]
        paths.extend(str(path) for path in selected)

    return paths

@st.cache_data(show_spinner="Preparing image features...")
def load_dataset_features(paths, model_version):
    _, feature_model, preprocess, _ = load_models()

    records = []

    for path_string in paths:
        path = Path(path_string)

        category = FOLDER_CATEGORIES.get(
            path.parent.name.lower()
        )

        if category is None:
            continue

        try:
            with Image.open(path) as source:
                image = source.convert("RGB")

            features = extract_features(
                image, feature_model, preprocess
            )

            records.append({
                "path": path_string,
                "category": category,
                "features": features
            })

        except Exception:
            continue

    return records

if not DATASET_PATH.exists():
    st.error(
        f"Dataset folder not found at:\n{DATASET_PATH}\n\n"
        "Make sure the dataset folder is beside app.py."
    )
    st.stop()

try:
    classifier, feature_model, preprocess, labels = load_models()
except Exception as error:
    st.error(
        "Could not load ResNet18. Check your internet "
        "connection and make sure torch and torchvision "
        "are installed."
    )
    st.exception(error)
    st.stop()

paths = get_image_paths(
    str(DATASET_PATH), MAX_IMAGES
)

if not paths:
    st.error(
        "No images found. Check that your dataset "
        "contains image files inside the category folders."
    )
    st.stop()

dataset_records = load_dataset_features(
    paths, "resnet18-v2"
)

if not dataset_records:
    st.error(
        "No image features could be loaded. "
        "Check the folder names and image files."
    )
    st.stop()

category_counts = {}

for record in dataset_records:
    category = record["category"]
    category_counts[category] = (
        category_counts.get(category, 0) + 1
    )

st.success(
    f"Ready! Loaded features for "
    f"{len(dataset_records)} images."
)

with st.expander("Dataset images by category"):
    for category, name in CATEGORY_NAMES.items():
        st.write(
            f"{name}: {category_counts.get(category, 0)} images"
        )


uploaded_file = st.file_uploader(
    "Upload an animal image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file is not None:
    try:
        query_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(query_image, width=250)

        category, confidence, predicted_label = predict_animal(
            query_image,
            classifier,
            preprocess,
            labels
        )

        if category is None:
            st.warning(
                "The model could not identify one of the "
                "Animals-10 categories. Try a clearer image."
            )
            st.stop()

        st.write(
            f"Detected category: **{CATEGORY_NAMES[category]}**"
        )

        st.caption(
            f"Model prediction: {predicted_label}"
        )

        candidates = [
            item for item in dataset_records
            if item["category"] == category
        ]

        if not candidates:
            st.warning(
                f"No {CATEGORY_NAMES[category].lower()} "
                "images are available in the loaded dataset."
            )
            st.stop()

        query_features = extract_features(
            query_image,
            feature_model,
            preprocess
        )

        feature_matrix = np.stack([
            item["features"] for item in candidates
        ])

        scores = cosine_similarity(
            query_features.reshape(1, -1),
            feature_matrix
        )[0]

        ranked_indices = np.argsort(scores)[::-1][
            :min(TOP_RESULTS, len(candidates))
        ]

        st.subheader(
            f"Similar {CATEGORY_NAMES[category]}"
        )

        st.caption(
            f"Searching {len(candidates)} images "
            "from this category."
        )

        columns = st.columns(len(ranked_indices))

        for column, index in zip(columns, ranked_indices):
            item = candidates[index]

            with column:
                with Image.open(item["path"]) as source:
                    result_image = source.convert("RGB")

                st.image(
                    result_image,
                    use_container_width=True
                )

                st.write(
                    f"Similarity: {scores[index] * 100:.1f}%"
                )

                st.caption(Path(item["path"]).name)

    except Exception as error:
        st.error(
            "An error occurred while processing the image."
        )
        st.exception(error)