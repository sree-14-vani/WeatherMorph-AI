# 🌦️ WeatherMorph AI

### Transform the Weather. Reimagine Your Photos.

WeatherMorph AI is a web-based image transformation application that lets users explore different weather conditions, seasons, and lighting effects on photographs. Built with Python and Streamlit, the application provides a simple interface for uploading images, applying visual transformations, comparing results, and downloading edited images.

## ✨ Features

- **Multiple Weather Effects:** Apply Winter, Rain, and Fog effects to photographs.
- **Seasonal Transformations:** Explore Summer, Spring, and Autumn visual effects.
- **Lighting Transformations:** Experiment with Day-to-Night and Night-to-Day effects.
- **Before-and-After Preview:** Compare the original photograph with the transformed result.
- **Single Image Processing:** Upload an individual image and select a transformation.
- **Batch Image Processing:** Process multiple images and download the results in a ZIP archive.
- **Download Results:** Save transformed images to your device.
- **Automatic Style Suggestions:** Get transformation suggestions based on the application’s supported logic.
- **User-Friendly Interface:** Navigate the application through a clean interface with a cream-and-maroon theme.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| OpenCV | Image processing |
| NumPy | Numerical and image-array operations |
| Pillow (PIL) | Image loading and manipulation |
| Git and GitHub | Version control and source-code hosting |
| Streamlit Community Cloud | Web application deployment |

## ⚙️ How It Works

1. Upload a photograph from your device.
2. Choose a supported weather, season, or lighting transformation.
3. Apply the selected effect.
4. Preview the original and transformed images.
5. Download the result or process additional photographs.

The current transformation pipeline uses image-processing techniques implemented in `weather_morph_core.py`. The application interface is implemented in `streamlit_app.py`.

## 🚀 Live Demo

**Try WeatherMorph AI:** [Open the deployed application](https://share.streamlit.io/)

> Replace this link with your application's actual public Streamlit URL.

## 📂 Project Structure

```text
WeatherMorph-AI/
│
├── app.py
├── streamlit_app.py
├── weather_morph_core.py
├── create_core.py
├── requirements.txt
├── README.md
└── .gitignore
```

- `app.py` — Original Gradio application.
- `streamlit_app.py` — Streamlit web application.
- `weather_morph_core.py` — Image transformation functions and supported modes.
- `create_core.py` — Utility script used during core-module creation.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Files excluded from version control.

## 💻 Run the Project Locally

### Prerequisites

- Python installed on your computer.
- Git installed on your computer.
- An internet connection for installing dependencies.

### 1. Clone the repository

```bash
git clone https://github.com/sree-14-vani/WeatherMorph-AI.git
```

### 2. Navigate to the project folder

```bash
cd WeatherMorph-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the application

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 📸 Supported Transformations

| Category | Available Effects |
|---|---|
| Weather | Winter, Rain, Fog |
| Seasons | Summer, Spring, Autumn |
| Lighting | Day-to-Night, Night-to-Day |

Results may vary depending on the input photograph and the selected transformation.

## 🎯 Project Objectives

- Develop an accessible web application for photo transformation.
- Explore image-processing techniques for simulating weather and lighting changes.
- Provide an interactive interface for individual and batch image processing.
- Demonstrate the integration of Python image-processing libraries with a web application.
- Deploy the application online for users to access.

## 🔮 Future Enhancements

- Integrate and validate a trained CycleGAN model for realistic seasonal image translation.
- Extend AI-based transformations to additional weather conditions.
- Improve visual realism and preserve important scene details.
- Add a before-and-after comparison slider.
- Optimize inference speed and memory usage.
- Add more evaluation methods for comparing transformation quality.

## ⚠️ Limitations

- The current transformations use image-processing techniques and may not produce photorealistic results for every image.
- Different photographs may produce different results.
- The trained CycleGAN model is not yet integrated into the currently deployed application.
- Application performance depends on the deployment environment and image size.

## 👩‍💻 Author

**Mitta Srivani**

- GitHub: [sree-14-vani](https://github.com/sree-14-vani)



⭐ If you find this project interesting, consider starring the repository on GitHub.
