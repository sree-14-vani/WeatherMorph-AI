
import os
import tempfile
import zipfile

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance, ImageStat

def arr(image):
    return np.asarray(image.convert("RGB")).astype(np.float32)


def pil(array):
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def blend(original, changed, intensity):
    strength = float(np.clip(intensity, 0, 1))
    return pil(original * (1 - strength) + changed * strength)


def sky_mask(image):
    """Estimate the sky region using color and image position."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    blue = (
        (hue >= 75) & (hue <= 135)
        & (sat >= 18) & (val >= 55)
    )

    pale = (sat < 65) & (val > 145)

    top_prior = np.clip(
        (np.linspace(1, 0, h)[:, None] - 0.22) / 0.48,
        0, 1
    )

    mask = (blue | pale).astype(np.float32) * top_prior
    mask = cv2.GaussianBlur(mask, (0, 0), 4)

    return np.clip(mask, 0, 1)[:, :, None]


def winter_transform(image, intensity):
    original = arr(image)
    height, width = original.shape[:2]

    winter = np.clip(
        original * np.array([0.84, 0.94, 1.12], dtype=np.float32) * 1.02 + 3,
        0, 255
    )

    hsv = cv2.cvtColor(
        original.astype(np.uint8), cv2.COLOR_RGB2HSV
    ).astype(np.float32)
    val = hsv[:, :, 2].copy()
    foliage = (
        (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 90)
        & (hsv[:, :, 1] >= 35)
    )
    ground = (
        (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 80)
    ).astype(np.float32)
    hsv[:, :, 1] = np.where(
        foliage, hsv[:, :, 1] * 0.30, hsv[:, :, 1] * 0.82
    )
    hsv[:, :, 2] = np.where(
        foliage, hsv[:, :, 2] * 1.16, hsv[:, :, 2]
    )
    pale = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2RGB
    ).astype(np.float32)
    foliage_mask = cv2.GaussianBlur(
        foliage.astype(np.float32), (0, 0), 1.5
    )[:, :, None]
    winter = winter * (1 - 0.78 * foliage_mask) + pale * (
        0.78 * foliage_mask
    )

    rng = np.random.default_rng(42)
    coarse_noise = rng.random(
        (max(2, height // 28), max(2, width // 28)),
        dtype=np.float32
    )
    snow_patches = cv2.resize(
        coarse_noise, (width, height), interpolation=cv2.INTER_CUBIC
    )
    snow_patches = cv2.GaussianBlur(snow_patches, (0, 0), 3)
    snow_patches = np.clip((snow_patches - 0.32) * 0.72, 0, 0.34)
    leaf_brightness = np.clip((val - 28) / 170, 0.2, 1)
    leaf_snow_mask = (
        foliage_mask[:, :, 0] * snow_patches * leaf_brightness
    )[:, :, None]
    snow_color = np.zeros_like(winter)
    snow_color[:] = [222, 234, 246]
    winter = winter * (1 - leaf_snow_mask) + snow_color * leaf_snow_mask

    lower_prior = np.clip(
        (np.linspace(0, 1, height, dtype=np.float32)[:, None] - 0.5) / 0.4,
        0, 1
    )[:, :, None]
    ground_mask = cv2.GaussianBlur(ground, (0, 0), 3)[:, :, None]
    ground_mask *= lower_prior
    ground_snow = np.clip(ground_mask * 0.38, 0, 0.24)
    winter = winter * (1 - ground_snow) + snow_color * (
        ground_snow
    )

    flakes = np.zeros((height, width), dtype=np.float32)
    for _ in range(max(12, height * width // 6500)):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        length = int(rng.choice([1, 1, 2, 3]))
        cv2.line(
            flakes, (x, y), (min(width - 1, x + 1), min(height - 1, y + length)),
            1.0, 1, cv2.LINE_AA
        )

    flakes = cv2.GaussianBlur(flakes, (0, 0), 0.35)[:, :, None]
    winter = winter * (1 - flakes * 0.48) + snow_color * (
        flakes * 0.48
    )
    return blend(original, winter, intensity)


def fog_transform(image, intensity):
    original = arr(image)
    height, width = original.shape[:2]

    distance = np.broadcast_to(
        np.linspace(1, 0.15, height, dtype=np.float32)[:, None],
        (height, width)
    )
    distance = cv2.GaussianBlur(distance, (0, 0), 14)[:, :, None]

    fog_color = np.zeros_like(original)
    fog_color[:] = [195, 205, 215]
    alpha = (0.025 + 0.17 * distance) * float(intensity)
    return pil(original * (1 - alpha) + fog_color * alpha)


def rain_transform(image, intensity):
    original = arr(image)
    height, width = original.shape[:2]

    rainy = original * np.array(
        [0.90, 0.93, 0.97], dtype=np.float32
    )

    overlay = np.zeros((height, width), dtype=np.float32)
    rng = np.random.default_rng(17)

    for _ in range(max(24, height * width // 1800)):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        length = int(rng.integers(10, 25))

        cv2.line(
            overlay,
            (x, y),
            (min(width - 1, x + 4), min(height - 1, y + length)),
            1.0,
            1,
            cv2.LINE_AA
        )

    streaks = cv2.GaussianBlur(overlay, (0, 0), 0.45)[:, :, None]
    rainy += streaks * np.array([45, 50, 58], dtype=np.float32)
    rainy_image = ImageEnhance.Color(pil(rainy)).enhance(0.92)
    return blend(original, arr(rainy_image), intensity)


def day_to_night(image, intensity):
    original = arr(image)
    strength = float(np.clip(intensity, 0, 1))

    if strength == 0:
        return image.convert("RGB")

    mask = sky_mask(image)
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    luminance = (
        0.299 * original[:, :, 0]
        + 0.587 * original[:, :, 1]
        + 0.114 * original[:, :, 2]
    )

    # Keep terrain detail, but shift the scene toward cool, low-light tones.
    scene = np.power(np.clip(original / 255.0, 0, 1), 1.45) * 255
    scene *= np.array([0.38, 0.48, 0.66], dtype=np.float32)
    scene = np.clip(scene, 0, 255)

    # Replace the daytime sky colors while retaining a subdued cloud silhouette.
    sun_core = (
        (luminance > 235) & (saturation < 55) & (mask[:, :, 0] > 0.5)
    ).astype(np.float32)
    flare_size = max(21, int(min(original.shape[:2]) * 0.16) | 1)
    sun_core = cv2.dilate(
        sun_core,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (flare_size, flare_size)
        )
    )
    sun_glow = cv2.GaussianBlur(
        sun_core, (0, 0), max(10, min(original.shape[:2]) * 0.10)
    )
    sun_glow = np.clip(sun_glow, 0, 1)[:, :, None]

    cloud_detail = np.clip((luminance - 65) / 150, 0, 1)
    cloud_detail = cv2.GaussianBlur(cloud_detail, (0, 0), 1.5)
    cloud_detail = (cloud_detail[:, :, None]) * (1 - sun_glow)
    night_sky = np.zeros_like(original)
    night_sky[:] = [5, 10, 25]
    cloud_tone = np.zeros_like(original)
    cloud_tone[:] = [38, 45, 59]
    night_sky += cloud_tone * (0.68 * cloud_detail)
    night_sky = night_sky * (1 - 0.85 * sun_glow) + np.array(
        [5, 10, 25], dtype=np.float32
    ) * (0.85 * sun_glow)
    night = scene * (1 - mask) + night_sky * mask

    # Stars belong only in unobstructed blue sky, never on clouds or sunlight.
    height, width = original.shape[:2]
    top_region = (
        np.linspace(1, 0, height, dtype=np.float32)[:, None] > 0.35
    )
    clear_sky = (
        (mask[:, :, 0] > 0.6)
        & top_region
        & (hue >= 78) & (hue <= 135)
        & (saturation >= 55)
        & (value >= 35) & (value <= 185)
    )
    star_positions = np.argwhere(clear_sky)
    if len(star_positions):
        rng = np.random.default_rng(23)
        count = min(35, max(8, len(star_positions) // 3500))
        chosen = rng.choice(
            len(star_positions),
            size=min(count, len(star_positions)),
            replace=False
        )
        for index in chosen:
            y, x = star_positions[index]
            cv2.circle(
                night, (int(x), int(y)), 1,
                (150, 170, 205), -1, cv2.LINE_AA
            )

    alpha = strength + (1 - strength) * 0.65 * mask
    alpha = np.maximum(alpha, 0.98 * sun_glow * mask)
    return pil(original * (1 - alpha) + night * alpha)


def night_to_day(image, intensity):
    original = arr(image)
    mask = sky_mask(image)

    lifted = np.sqrt(np.clip(original / 255, 0, 1)) * 255
    lifted = np.clip(lifted * 1.12, 0, 255)

    lifted[:, :, 0] = np.clip(
        lifted[:, :, 0] * 1.06 + 5, 0, 255
    )
    lifted[:, :, 1] = np.clip(
        lifted[:, :, 1] * 1.04 + 3, 0, 255
    )

    sky = lifted.copy()
    sky[:, :, 0] = np.clip(
        sky[:, :, 0] * 0.78 + 35, 0, 255
    )
    sky[:, :, 1] = np.clip(
        sky[:, :, 1] * 0.90 + 35, 0, 255
    )
    sky[:, :, 2] = np.clip(
        sky[:, :, 2] * 1.10 + 20, 0, 255
    )

    day = lifted * (1 - mask) + sky * mask
    return blend(original, day, intensity)


def summer_transform(image, intensity):
    original = arr(image)
    summer = original.copy()

    summer[:, :, 0] = np.clip(
        summer[:, :, 0] * 1.07 + 4, 0, 255
    )
    summer[:, :, 1] = np.clip(
        summer[:, :, 1] * 1.04 + 2, 0, 255
    )

    result = ImageEnhance.Color(pil(summer)).enhance(1.15)
    result = ImageEnhance.Contrast(result).enhance(1.04)

    return blend(original, arr(result), intensity)


def spring_transform(image, intensity):
    original = arr(image)

    hsv = cv2.cvtColor(
        original.astype(np.uint8), cv2.COLOR_RGB2HSV
    ).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    foliage = (
        (hue >= 20) & (hue <= 88)
        & (saturation >= 30) & (hsv[:, :, 2] >= 30)
    )
    fresh_hue = np.clip(
        hue + np.maximum(0, 46 - hue) * 0.55, 0, 179
    )
    hsv[:, :, 0] = np.where(foliage, fresh_hue, hue)
    hsv[:, :, 1] = np.where(foliage, saturation * 1.12, saturation)
    hsv[:, :, 2] = np.where(
        foliage, hsv[:, :, 2] * 1.05, hsv[:, :, 2]
    )
    spring = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2RGB
    ).astype(np.float32)
    spring = np.clip(spring * 1.015 + 1, 0, 255)

    return blend(original, spring, intensity)


def autumn_transform(image, intensity):
    original = arr(image)

    hsv = cv2.cvtColor(
        original.astype(np.uint8), cv2.COLOR_RGB2HSV
    ).astype(np.float32)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    foliage = (
        (hue >= 18) & (hue <= 82)
        & (sat >= 35) & (val >= 30)
    )

    autumn_hue = 13 + (hue - 18) * 0.22
    hsv[:, :, 0] = np.where(foliage, autumn_hue, hue)
    hsv[:, :, 1] = np.where(foliage, sat * 1.18 + 8, sat)
    hsv[:, :, 2] = np.where(foliage, val * 0.92, val)

    autumn_foliage = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2RGB
    ).astype(np.float32)

    foliage_mask = cv2.GaussianBlur(
        foliage.astype(np.float32), (0, 0), 2.5
    )[:, :, None]

    warm_scene = original.copy()
    warm_scene[:, :, 0] = np.clip(
        warm_scene[:, :, 0] * 1.04 + 3, 0, 255
    )
    warm_scene[:, :, 1] = np.clip(
        warm_scene[:, :, 1] * 1.01, 0, 255
    )
    warm_scene[:, :, 2] = np.clip(
        warm_scene[:, :, 2] * 0.93, 0, 255
    )

    autumn = (
        autumn_foliage * foliage_mask
        + warm_scene * (1 - foliage_mask)
    )

    return blend(original, autumn, intensity)


MODES = [
    "Winter",
    "Rain",
    "Fog",
    "Day-to-Night",
    "Night-to-Day",
    "Summer",
    "Spring",
    "Autumn"
]


def apply_transformation(image, mode, intensity):
    functions = {
        "Rain": rain_transform,
        "Fog": fog_transform,
        "Day-to-Night": day_to_night,
        "Night-to-Day": night_to_day,
        "Summer": summer_transform,
        "Spring": spring_transform,
        "Autumn": autumn_transform,
        "Winter": winter_transform,
    }

    return functions[mode](image, intensity)


def suggest_style(image):
    if image is None:
        return "Upload a photo to receive a suggestion."

    brightness = sum(
        ImageStat.Stat(image.convert("RGB")).mean
    ) / 3

    if brightness > 175:
        return "Suggestion: try Day-to-Night, Fog, or Winter."

    if brightness < 75:
        return "Suggestion: try Night-to-Day."

    return "Suggestion: try Rain, Fog, Summer, Spring, or Autumn."


def transform(image, mode, intensity):
    if image is None:
        raise gr.Error("Please upload a photo first.")

    result = apply_transformation(image, mode, intensity)
    original_pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    result_pixels = np.asarray(result.convert("RGB"), dtype=np.int16)
    mean_change = float(np.abs(original_pixels - result_pixels).mean())
    status = (
        f"{mode} applied at {float(intensity):.0%} intensity. "
        f"Average pixel change: {mean_change:.1f}/255."
    )
    if intensity > 0 and mean_change < 0.5:
        raise gr.Error(
            "The selected effect made almost no visible change. "
            "Increase Effect Strength and transform again."
        )
    return result, suggest_style(image), status


def batch_transform(files, mode, intensity):
    if not files:
        raise gr.Error("Please upload one or more photos.")

    output_zip = os.path.join(
        tempfile.gettempdir(), "WeatherMorph_Results.zip"
    )

    success = 0
    failed = []

    with zipfile.ZipFile(
        output_zip, "w", zipfile.ZIP_DEFLATED
    ) as archive:

        for index, path in enumerate(files, start=1):
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")

                result = apply_transformation(
                    image, mode, intensity
                )

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp:
                    temp_path = temp.name

                try:
                    result.save(temp_path, quality=95)
                    archive.write(
                        temp_path,
                        arcname=f"transformed_{index:05d}.jpg"
                    )
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                success += 1

            except Exception as exc:
                failed.append(os.path.basename(path))
                print(f"Could not process {path}: {exc}")

    if success == 0:
        raise gr.Error(
            "No images could be processed. Check the terminal for errors."
        )

    status = f"Processed {success} image(s)."

    if failed:
        status += (
            f" Skipped {len(failed)}: "
            + ", ".join(failed[:10])
        )

    return output_zip, status


APP_CSS = """
html, body {
    color-scheme: light !important;
    background: #f5f0e5 !important;
    color: #353a34 !important;
}
.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    --color-accent: #702b3b !important;
    --color-accent-soft: #e9ddd3 !important;
    --button-primary-background-fill: #702b3b !important;
    --button-primary-background-fill-hover: #57202f !important;
    --button-primary-text-color: #fffaf0 !important;
    --slider-color: #702b3b !important;
    color: #353a34 !important;
    background:
        radial-gradient(ellipse at 12% 0%, rgba(135, 148, 129, .16), transparent 34%),
        radial-gradient(ellipse at 95% 16%, rgba(112, 43, 59, .08), transparent 30%),
        #f5f0e5 !important;
}
.gradio-container .block {
    color: #353a34 !important;
    background-color: #fffaf0 !important;
    border-color: #d8d2c4 !important;
}
.gradio-container .wrap,
.gradio-container .secondary-wrap,
.gradio-container [role="listbox"],
.gradio-container [role="option"],
.gradio-container [role="tabpanel"] {
    color: #353a34 !important;
    background-color: #fffaf0 !important;
}
.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"] {
    color: #57202f !important;
    background-color: #e9ddd3 !important;
}
.gradio-container label,
.gradio-container label span,
.gradio-container .block-info {
    color: #3e2027 !important;
    -webkit-text-fill-color: #3e2027 !important;
    font-weight: 700 !important;
}
.gradio-container .block label,
.gradio-container .wrap label,
.gradio-container .wrap label span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span {
    color: #3e2027 !important;
    -webkit-text-fill-color: #3e2027 !important;
    background-color: #eee9dc !important;
    font-weight: 700 !important;
}
.gradio-container .block label {
    font-size: 15px !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #596258 !important;
    opacity: 1;
}
.gradio-container [role="tab"] {
    color: #596258 !important;
    border-color: #d8d2c4 !important;
}
.gradio-container [role="tab"][aria-selected="true"] {
    color: #702b3b !important;
    border-color: #702b3b !important;
}
.gradio-container .tabs,
.gradio-container [role="tablist"] {
    border-color: #d8d2c4 !important;
}
.hero {
    padding: 34px 38px;
    margin: 12px 0 24px;
    border: 1px solid #d8d2c4;
    border-radius: 24px;
    background: linear-gradient(120deg, #fffaf0, #eee9dc);
    box-shadow: 0 20px 60px rgba(53, 58, 52, .09);
}
.hero-kicker {
    color: #702b3b;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
}
.hero h1 {
    margin: 10px 0 8px;
    color: #353a34;
    font-size: clamp(32px, 5vw, 52px);
    line-height: 1.08;
}
.hero p {
    max-width: 720px;
    margin: 0;
    color: #596258;
    font-size: 16px;
    line-height: 1.65;
}
.gradio-container .tabs {
    border-bottom: 1px solid #d8d2c4;
}
.gradio-container button.primary {
    min-height: 54px;
    border: 0 !important;
    border-radius: 14px !important;
    background: linear-gradient(100deg, #7a3545, #57202f) !important;
    color: #fffaf0 !important;
    box-shadow: 0 10px 28px rgba(87, 32, 47, .2);
    font-size: 16px !important;
    font-weight: 700 !important;
}
.gradio-container button.primary:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}
.gradio-container input, .gradio-container textarea,
.gradio-container [role="combobox"] {
    border-radius: 12px !important;
    color: #353a34 !important;
    background-color: #fffaf0 !important;
    border-color: #b9b8aa !important;
}
.result-note {
    padding: 14px 18px;
    border: 1px solid #d8d2c4;
    border-radius: 14px;
    background: #eee9dc;
    color: #353a34;
}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="green", secondary_hue="orange", neutral_hue="slate"
).set(
    body_background_fill="#f5f0e5",
    body_background_fill_dark="#f5f0e5",
    body_text_color="#353a34",
    body_text_color_dark="#353a34",
    body_text_color_subdued="#596258",
    body_text_color_subdued_dark="#596258",
    background_fill_primary="#fffaf0",
    background_fill_primary_dark="#fffaf0",
    background_fill_secondary="#eee9dc",
    background_fill_secondary_dark="#eee9dc",
    color_accent="#702b3b",
    color_accent_soft="#e9ddd3",
    color_accent_soft_dark="#e9ddd3",
    block_background_fill="#fffaf0",
    block_background_fill_dark="#fffaf0",
    block_border_color="#d8d2c4",
    block_border_color_dark="#d8d2c4",
    block_label_background_fill="#eee9dc",
    block_label_background_fill_dark="#eee9dc",
    block_label_text_color="#353a34",
    block_label_text_color_dark="#353a34",
    block_title_text_color="#353a34",
    block_title_text_color_dark="#353a34",
    block_info_text_color="#596258",
    block_info_text_color_dark="#596258",
    input_background_fill="#fffaf0",
    input_background_fill_dark="#fffaf0",
    input_border_color="#b9b8aa",
    input_border_color_dark="#b9b8aa",
    button_primary_background_fill="#702b3b",
    button_primary_background_fill_dark="#702b3b",
    button_primary_background_fill_hover="#57202f",
    button_primary_background_fill_hover_dark="#57202f",
    button_primary_text_color="#fffaf0",
    button_primary_text_color_dark="#fffaf0",
    slider_color="#702b3b",
    slider_color_dark="#702b3b",
    link_text_color="#702b3b",
    link_text_color_dark="#702b3b",
    link_text_color_hover="#57202f",
    link_text_color_hover_dark="#57202f",
)

with gr.Blocks(
    title="WeatherMorph AI",
    theme=APP_THEME,
    css=APP_CSS,
) as demo:

    gr.HTML(
        '<section class="hero">'
        '<div class="hero-kicker">WeatherMorph AI &nbsp; / &nbsp; Photo studio</div>'
        '<h1>Give every photo a new atmosphere.</h1>'
        '<p>Explore seasonal looks, changing skies, and weather effects. '
        'Upload a photo, choose a transformation, and compare the result.</p>'
        '</section>'
    )

    with gr.Tab("Single Photo"):
        with gr.Row():
            input_image = gr.Image(
                type="pil", label="01  ·  Original Photo", height=470
            )
            output_image = gr.Image(
                type="pil", label="02  ·  Transformed Photo", height=470
            )

        mode = gr.Dropdown(
            choices=MODES,
            value=MODES[0],
            label="Choose your atmosphere"
        )

        intensity = gr.Slider(
            0, 1, value=0.8, step=0.05,
            label="Effect strength"
        )

        run_button = gr.Button(
            "Transform photo", variant="primary"
        )

        result_status = gr.Markdown(
            elem_classes=["result-note"]
        )

        suggestion = gr.Textbox(
            label="Photo suggestion", interactive=False
        )

        run_button.click(
            transform,
            [input_image, mode, intensity],
            [output_image, suggestion, result_status]
        )

    with gr.Tab("Batch Photos"):
        gr.Markdown(
            "Apply the same look and strength to multiple photos, "
            "then download the results together."
        )
        files = gr.File(
            file_count="multiple",
            file_types=["image"],
            type="filepath",
            label="Upload Multiple Photos"
        )

        batch_mode = gr.Dropdown(
            choices=MODES,
            value=MODES[0],
            label="Transformation"
        )

        batch_intensity = gr.Slider(
            0, 1, value=0.8, step=0.05,
            label="Effect Intensity"
        )

        batch_button = gr.Button(
            "Process Photos", variant="primary"
        )

        zip_output = gr.File(
            label="Download Transformed Photos (ZIP)"
        )

        batch_status = gr.Textbox(
            label="Processing Status", interactive=False
 
        )

        batch_button.click(
            batch_transform,
            [files, batch_mode, batch_intensity],
            [zip_output, batch_status]
        )

    gr.Markdown(
        "**About these effects:** All effects use local image "
        "processing rather than a generative image model. Winter adds "
        "cooling and light snowfall; Spring and Autumn recolor detected "
        "foliage, so seasonal results are strongest in photos with trees "
        "or other green plants. "
        "Sky detection is approximate. "
        "Batch capacity depends on available memory, storage and "
        "processing time.",
        elem_classes=["result-note"]
    )


demo.queue().launch(
    inbrowser=True,
    server_name="127.0.0.1",
    server_port=7860,
    show_error=True,
)