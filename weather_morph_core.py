
import os
import tempfile
import zipfile

import cv2
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
        raise ValueError("Please upload a photo first.")

    result = apply_transformation(image, mode, intensity)
    original_pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    result_pixels = np.asarray(result.convert("RGB"), dtype=np.int16)
    mean_change = float(np.abs(original_pixels - result_pixels).mean())
    status = (
        f"{mode} applied at {float(intensity):.0%} intensity. "
        f"Average pixel change: {mean_change:.1f}/255."
    )
    if intensity > 0 and mean_change < 0.5:
        raise ValueError(
            "The selected effect made almost no visible change. "
            "Increase Effect Strength and transform again."
        )
    return result, suggest_style(image), status


def batch_transform(files, mode, intensity):
    if not files:
        raise ValueError("Please upload one or more photos.")

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
        raise ValueError(
            "No images could be processed. Check the terminal for errors."
        )

    status = f"Processed {success} image(s)."

    if failed:
        status += (
            f" Skipped {len(failed)}: "
            + ", ".join(failed[:10])
        )

    return output_zip, status


