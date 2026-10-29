
import io
import zipfile

import streamlit as st
from PIL import Image

from weather_morph_core import (
    apply_transformation,
    MODES,
    suggest_style,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="WeatherMorph AI | Photo Studio",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CUSTOM DESIGN — MATCHING YOUR GRADIO APP
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: #f5f0e5 !important;
        color: #353a34 !important;
    }

    [data-testid="stHeader"] {
        background: #f5f0e5 !important;
    }

    [data-testid="stMain"] {
        background:
            radial-gradient(
                ellipse at 12% 0%,
                rgba(135, 148, 129, .16),
                transparent 34%
            ),
            radial-gradient(
                ellipse at 95% 16%,
                rgba(112, 43, 59, .08),
                transparent 30%
            ),
            #f5f0e5 !important;
    }

    .block-container {
        max-width: 1440px;
        padding: 2rem 2.2rem 3rem;
    }

    /* Hero section */
    .hero {
        padding: 38px;
        margin: 8px 0 28px;
        border: 1px solid #d8d2c4;
        border-radius: 24px;
        background: linear-gradient(
            120deg,
            #fffaf0,
            #eee9dc
        );
        box-shadow: 0 20px 60px rgba(53, 58, 52, .09);
    }

    .hero-kicker {
        color: #702b3b;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: 12px 0 10px;
        color: #353a34;
        font-size: clamp(32px, 4vw, 52px);
        line-height: 1.1;
        font-weight: 800;
    }

    .hero p {
        max-width: 740px;
        margin: 0;
        color: #596258;
        font-size: 16px;
        line-height: 1.7;
    }

    /* Section headings */
    h2, h3 {
        color: #353a34 !important;
        font-weight: 750 !important;
    }

    p, li, label, span {
        color: inherit;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #353a34;
    }

    /* Main content cards */
    [data-testid="stFileUploader"],
    [data-testid="stImage"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #d8d2c4 !important;
    }

    [data-testid="stFileUploader"] {
        padding: 12px;
        border: 1px solid #d8d2c4;
        border-radius: 16px;
        background: #fffaf0;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #fffaf0 !important;
        border: 1px dashed #b9b8aa !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #353a34 !important;
    }

    /* Inputs and dropdowns */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: #fffaf0 !important;
        border-color: #b9b8aa !important;
        border-radius: 12px !important;
        color: #353a34 !important;
    }

    [data-baseweb="select"] * {
        color: #353a34 !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] {
        color: #702b3b !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        min-height: 48px;
        width: 100%;
        border: 1px solid #702b3b !important;
        border-radius: 12px !important;
        background: linear-gradient(
            100deg,
            #7a3545,
            #57202f
        ) !important;
        color: #fffaf0 !important;
        font-size: 15px !important;
        font-weight: 750 !important;
        transition: transform .15s ease, filter .15s ease;
    }

    .stButton > button *,
    .stDownloadButton > button * {
        color: #fffaf0 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
    }

    .stButton > button:focus,
    .stDownloadButton > button:focus {
        box-shadow: 0 0 0 2px #e9ddd3 !important;
    }

    /* Tabs */
    [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid #d8d2c4;
    }

    [data-baseweb="tab"] {
        background: #eee9dc !important;
        border-radius: 10px 10px 0 0;
        color: #596258 !important;
    }

    [data-baseweb="tab"] * {
        color: #596258 !important;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background: #fffaf0 !important;
        border-bottom: 3px solid #702b3b !important;
    }

    [data-baseweb="tab"][aria-selected="true"] * {
        color: #702b3b !important;
        font-weight: 800 !important;
    }

    /* Result notes and information cards */
    .result-note {
        padding: 16px 18px;
        margin: 10px 0;
        border: 1px solid #d8d2c4;
        border-radius: 14px;
        background: #eee9dc;
        color: #353a34;
        line-height: 1.65;
    }

    .photo-label {
        padding: 10px 14px;
        margin-bottom: 10px;
        border: 1px solid #d8d2c4;
        border-radius: 10px;
        background: #fffaf0;
        color: #3e2027;
        font-size: 14px;
        font-weight: 800;
    }

    .footer {
        margin-top: 36px;
        padding-top: 18px;
        border-top: 1px solid #d8d2c4;
        color: #596258;
        font-size: 13px;
        text-align: center;
    }

    /* Hide Streamlit's default decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .hero {
            padding: 24px;
            border-radius: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">
            WeatherMorph AI &nbsp; / &nbsp; Photo Studio
        </div>
        <h1>Give every photo a new atmosphere.</h1>
        <p>
            Explore seasonal looks, changing skies, and weather effects.
            Upload a photo, choose a transformation, and compare your
            original image with the result.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SINGLE PHOTO / BATCH PHOTOS
# --------------------------------------------------
single_tab, batch_tab = st.tabs(
    ["Single Photo", "Batch Photos"]
)

with single_tab:
    st.subheader("Transform a photo")
    st.write(
        "Choose an image and customize the atmosphere."
    )

    uploaded_file = st.file_uploader(
        "01 · Upload your original photo",
        type=["jpg", "jpeg", "png", "webp"],
        key="single_upload",
    )

    col_settings, col_info = st.columns(
        [1.0, 1.0],
        gap="large",
    )

    with col_settings:
        st.markdown("### Transformation settings")

        mode = st.selectbox(
            "Choose your atmosphere",
            MODES,
            index=MODES.index("Winter"),
        )

        intensity = st.slider(
            "Effect strength",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            format="%.2f",
        )

        transform_button = st.button(
            "Transform photo",
            type="primary",
            use_container_width=True,
        )

    with col_info:
        st.markdown("### Photo suggestion")

        if uploaded_file is not None:
            try:
                suggestion_image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.markdown(
                    f'<div class="result-note">'
                    f'{suggest_style(suggestion_image)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                st.info("Please upload a valid image.")
        else:
            st.markdown(
                """
                <div class="result-note">
                    Upload a photo to receive a transformation
                    suggestion.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="result-note">
                <b>Available looks</b><br>
                Winter · Rain · Fog · Day-to-Night ·
                Night-to-Day · Summer · Spring · Autumn
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Process only when the button is clicked
    if transform_button:
        if uploaded_file is None:
            st.error("Please upload a photo first.")
        else:
            try:
                with Image.open(uploaded_file) as source:
                    original_image = source.convert("RGB")

                with st.spinner(
                    f"Applying {mode} transformation..."
                ):
                    result_image = apply_transformation(
                        original_image,
                        mode,
                        intensity,
                    )

                st.session_state["wm_original"] = original_image
                st.session_state["wm_result"] = result_image
                st.session_state["wm_mode"] = mode
                st.session_state["wm_intensity"] = intensity

                st.success("Transformation completed.")

            except Exception as exc:
                st.error(
                    f"Could not transform the image: {exc}"
                )

    # Show the latest result
    if "wm_result" in st.session_state:
        st.divider()
        st.subheader("Your transformation")

        original = st.session_state["wm_original"]
        result = st.session_state["wm_result"]

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(
                '<div class="photo-label">01 · Original Photo</div>',
                unsafe_allow_html=True,
            )
            st.image(original, width="stretch")

        with right:
            st.markdown(
                '<div class="photo-label">02 · Transformed Photo</div>',
                unsafe_allow_html=True,
            )
            st.image(result, width="stretch")

        output_buffer = io.BytesIO()
        result.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        st.download_button(
            "Download transformed photo",
            data=output_buffer.getvalue(),
            file_name=(
                f"WeatherMorph_"
                f"{st.session_state['wm_mode'].replace('-', '_')}.png"
            ),
            mime="image/png",
            use_container_width=True,
        )

        original_pixels = list(original.convert("RGB").getdata())
        result_pixels = list(result.convert("RGB").getdata())

        if len(original_pixels) == len(result_pixels):
            mean_change = sum(
                sum(abs(a - b) for a, b in zip(p, q)) / 3
                for p, q in zip(original_pixels, result_pixels)
            ) / len(original_pixels)

            st.markdown(
                f"""
                <div class="result-note">
                    <b>Transformation:</b>
                    {st.session_state['wm_mode']}<br>
                    <b>Effect strength:</b>
                    {st.session_state['wm_intensity']:.0%}<br>
                    <b>Average pixel difference:</b>
                    {mean_change:.2f}/255
                </div>
                """,
                unsafe_allow_html=True,
            )

with batch_tab:
    st.subheader("Batch photo studio")
    st.write(
        "Apply the same transformation to multiple photos "
        "and download all results in one ZIP file."
    )

    batch_files = st.file_uploader(
        "Upload multiple photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    batch_mode = st.selectbox(
        "Batch transformation",
        MODES,
        index=MODES.index("Winter"),
        key="batch_mode",
    )

    batch_intensity = st.slider(
        "Batch effect strength",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        key="batch_intensity",
    )

    if st.button(
        "Process photos",
        type="primary",
        use_container_width=True,
    ):
        if not batch_files:
            st.error("Please upload one or more photos.")
        else:
            zip_buffer = io.BytesIO()
            success_count = 0
            failed_files = []

            progress = st.progress(0)
            status = st.empty()

            with zipfile.ZipFile(
                zip_buffer,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                total = len(batch_files)

                for index, file in enumerate(batch_files, start=1):
                    try:
                        with Image.open(file) as source:
                            image = source.convert("RGB")

                        transformed = apply_transformation(
                            image,
                            batch_mode,
                            batch_intensity,
                        )

                        image_buffer = io.BytesIO()
                        transformed.save(
                            image_buffer,
                            format="JPEG",
                            quality=95,
                        )

                        archive.writestr(
                            f"transformed_{index:03d}.jpg",
                            image_buffer.getvalue(),
                        )

                        success_count += 1

                    except Exception as exc:
                        failed_files.append(file.name)
                        print(
                            f"Could not process {file.name}: {exc}"
                        )

                    progress.progress(index / total)
                    status.write(
                        f"Processed {index} of {total} photos..."
                    )

            if success_count:
                st.success(
                    f"Successfully processed "
                    f"{success_count} photo(s)."
                )

                st.download_button(
                    "Download all results (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="WeatherMorph_Results.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

            if failed_files:
                st.warning(
                    "Could not process: "
                    + ", ".join(failed_files[:10])
                )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        WEATHERMORPH AI · PHOTO STUDIO<br>
        Explore new atmospheres, one photo at a time.
    </div>
    """,
    unsafe_allow_html=True,
)
