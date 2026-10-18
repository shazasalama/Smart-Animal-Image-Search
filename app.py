import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

st.set_page_config(
    page_title="Smart Animal Image Search",
    page_icon="🐾",
    layout="wide",
)

st.title("🐾 Smart Animal Image Search")
st.write(
    "Content-Based Image Retrieval (CBIR): upload an image and retrieve "
    "the most visually similar images from the Animals-10 dataset."
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset"

MODEL_NAME = "facebook/dinov2-small"

MAX_IMAGES_PER_CATEGORY = 50
DEFAULT_TOP_K = 10
MAX_TOP_K = 20

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
    "scoiattolo": "squirrel",
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
    "squirrel": "Squirrels",
}

DEEP_FEATURE_WEIGHT = 0.95
COLOR_FEATURE_WEIGHT = 0.05
COLOR_BINS_PER_CHANNEL = 16

INDEX_VERSION = "dinov2-small-cls-patches-v1"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

@st.cache_resource(show_spinner="Loading DINOv2 Small...")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    return model, processor

def color_histogram(image):
    image = image.convert("RGB").resize((128, 128))
    array = np.asarray(image, dtype=np.uint8)

    histograms = []

    for channel in range(3):
        hist, _ = np.histogram(
            array[:, :, channel],
            bins=COLOR_BINS_PER_CHANNEL,
            range=(0, 256),
        )

        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-12
        histograms.append(hist)

    vector = np.concatenate(histograms).astype(np.float32)
    vector /= np.linalg.norm(vector) + 1e-12

    return vector

def extract_features(image, model, processor):
    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

        tokens = outputs.last_hidden_state

        cls_vector = F.normalize(
            tokens[:, 0, :],
            p=2,
            dim=1,
        )

        patch_vector = F.normalize(
            tokens[:, 1:, :].mean(dim=1),
            p=2,
            dim=1,
        )

        deep_vector = torch.cat(
            [
                cls_vector * np.sqrt(0.60),
                patch_vector * np.sqrt(0.40),
            ],
            dim=1,
        )

        deep_vector = F.normalize(
            deep_vector,
            p=2,
            dim=1,
        )[0].cpu().numpy().astype(np.float32)

    color_vector = color_histogram(image)

    fused_vector = np.concatenate(
        [
            deep_vector * np.sqrt(DEEP_FEATURE_WEIGHT),
            color_vector * np.sqrt(COLOR_FEATURE_WEIGHT),
        ]
    ).astype(np.float32)

    fused_vector /= np.linalg.norm(fused_vector) + 1e-12

    return fused_vector, deep_vector

@st.cache_data(show_spinner=False)
def get_image_paths(dataset_path, max_per_category, extensions):
    root = Path(dataset_path)

    if not root.is_dir():
        return []

    paths = []

    for folder_name in FOLDER_CATEGORIES:
        category_folder = root / folder_name

        if not category_folder.is_dir():
            continue

        category_images = sorted(
            path
            for path in category_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
        )

        paths.extend(
            str(path)
            for path in category_images[:max_per_category]
        )

    return paths


@st.cache_data(
    show_spinner=(
        "Building the DINOv2 image-search index. "
        "The first run may take a few minutes..."
    )
)
def build_image_index(path_strings, model_version):
    feature_model, processor = load_model()

    records = []
    progress = st.progress(0, text="Indexing dataset images...")
    total = len(path_strings)

    for i, path_string in enumerate(path_strings):
        path = Path(path_string)
        category = FOLDER_CATEGORIES.get(path.parent.name.lower())

        if category is not None:
            try:
                with Image.open(path) as image:
                    fused_vector, deep_vector = extract_features(
                        image,
                        feature_model,
                        processor,
                    )

                records.append(
                    {
                        "path": path_string,
                        "category": category,
                        "features": fused_vector,
                        "deep_features": deep_vector,
                    }
                )

            except Exception:
                pass

        if total:
            progress.progress(
                (i + 1) / total,
                text=f"Indexing image {i + 1}/{total}...",
            )

    progress.empty()

    return records

def retrieve(
    query_vector,
    query_deep_vector,
    records,
    top_k,
    category_filter=None,
):
    candidates = [
        record
        for record in records
        if category_filter is None
        or record["category"] == category_filter
    ]

    if not candidates:
        return [], 0

    fused_matrix = np.stack(
        [record["features"] for record in candidates]
    )

    deep_matrix = np.stack(
        [record["deep_features"] for record in candidates]
    )

    fused_scores = fused_matrix @ query_vector
    deep_scores = deep_matrix @ query_deep_vector

    scores = 0.90 * deep_scores + 0.10 * fused_scores

    ranked_indices = np.argsort(scores)[::-1]
    results = []

    for index in ranked_indices:
        record = candidates[index]

        results.append(
            {
                "path": record["path"],
                "category": record["category"],
                "score": float(scores[index]),
            }
        )

        if len(results) >= top_k:
            break

    return results, len(candidates)

def category_metrics(results, relevant_category, records):
    if not results or relevant_category is None:
        return None, None

    relevant_retrieved = sum(
        result["category"] == relevant_category
        for result in results
    )

    precision_at_k = relevant_retrieved / len(results)

    total_relevant = sum(
        record["category"] == relevant_category
        for record in records
    )

    recall_at_k = (
        relevant_retrieved / total_relevant
        if total_relevant
        else 0.0
    )

    return precision_at_k, recall_at_k

if not DATASET_PATH.is_dir():
    st.error(
        f"Dataset folder not found:\n\n`{DATASET_PATH}`\n\n"
        "The dataset folder must be in the same repository directory as app.py."
    )
    st.stop()

missing_folders = [
    folder
    for folder in FOLDER_CATEGORIES
    if not (DATASET_PATH / folder).is_dir()
]

if missing_folders:
    st.error(
        "These required category folders are missing: "
        + ", ".join(missing_folders)
    )
    st.stop()

try:
    feature_model, processor = load_model()

except Exception as error:
    st.error(
        "Could not load DINOv2 Small. Check transformers, torch, "
        "and internet access for downloading the pretrained model."
    )
    st.exception(error)
    st.stop()

paths = get_image_paths(
    str(DATASET_PATH),
    MAX_IMAGES_PER_CATEGORY,
    tuple(sorted(EXTENSIONS)),
)

if not paths:
    st.error("No supported image files were found in the dataset folders.")
    st.stop()

records = build_image_index(paths, INDEX_VERSION)

if not records:
    st.error("No image features could be created. Check the dataset images.")
    st.stop()

counts = {
    category: 0
    for category in CATEGORY_NAMES
}

for record in records:
    counts[record["category"]] += 1

m1, m2, m3 = st.columns(3)

m1.metric("Indexed images", len(records))
m2.metric("Categories", sum(count > 0 for count in counts.values()))
m3.metric("Max images per category", MAX_IMAGES_PER_CATEGORY)

with st.expander("Dataset summary"):
    for category, name in CATEGORY_NAMES.items():
        st.write(f"**{name}:** {counts[category]} images")

st.subheader("Search")

search_mode = st.radio(
    "Where should the system search?",
    ["All categories", "One category"],
    horizontal=True,
)

category_filter = None

if search_mode == "One category":
    category_names = {
        CATEGORY_NAMES[category]: category
        for category in CATEGORY_NAMES
        if counts[category] > 0
    }

    selected_name = st.selectbox(
        "Select a category",
        list(category_names),
    )

    category_filter = category_names[selected_name]

top_k = st.slider(
    "Number of results to retrieve (Top-K)",
    min_value=1,
    max_value=MAX_TOP_K,
    value=DEFAULT_TOP_K,
)

uploaded_file = st.file_uploader(
    "Upload a query image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    try:
        query_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Query image")
        st.image(query_image, width=280)

        query_vector, query_deep_vector = extract_features(
            query_image,
            feature_model,
            processor,
        )

        results, candidate_count = retrieve(
            query_vector,
            query_deep_vector,
            records,
            top_k,
            category_filter=category_filter,
        )

        if not results:
            st.warning(
                "No results were found. Try another image "
                "or change the category filter."
            )
            st.stop()

        st.subheader("Retrieved images — ranked by visual similarity")

        st.caption(
            f"Showing {len(results)} results from {candidate_count} "
            "candidate images. Rank #1 has the highest score."
        )

        for start in range(0, len(results), 5):
            batch = results[start:start + 5]
            columns = st.columns(len(batch))

            for offset, (column, result) in enumerate(zip(columns, batch)):
                with column:
                    with Image.open(result["path"]) as image:
                        st.image(
                            image.convert("RGB"),
                            use_container_width=True,
                        )

                    st.write(f"**Rank #{start + offset + 1}**")
                    st.write(
                        f"Similarity score: {result['score']:.4f}"
                    )
                    st.caption(CATEGORY_NAMES[result["category"]])
                    st.caption(Path(result["path"]).name)

        st.info(
            "Similarity scores are mathematical measures, not guaranteed "
            "percentages of human-perceived similarity. This version uses "
            "DINOv2 Small visual features combined with color features."
        )

        st.divider()
        st.subheader("Optional IR evaluation")

        st.write(
            "Choose the query image's known category to calculate "
            "category-level Precision@K and Recall@K. These metrics "
            "measure category matching, not fine-grained appearance similarity."
        )

        evaluation_options = ["Not specified"] + [
            CATEGORY_NAMES[category]
            for category in CATEGORY_NAMES
        ]

        known_query_category_name = st.selectbox(
            "Known category of the query image",
            evaluation_options,
        )

        if known_query_category_name != "Not specified":
            name_to_category = {
                name: category
                for category, name in CATEGORY_NAMES.items()
            }

            known_query_category = name_to_category[
                known_query_category_name
            ]

            precision, recall = category_metrics(
                results,
                known_query_category,
                records,
            )

            eval1, eval2 = st.columns(2)

            eval1.metric(
                f"Precision@{len(results)}",
                f"{precision * 100:.1f}%",
            )

            eval2.metric(
                f"Recall@{len(results)}",
                f"{recall * 100:.1f}%",
            )

    except Exception as error:
        st.error("An error occurred while processing the query image.")
        st.exception(error)
