import streamlit as st
import os
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from torchvision import transforms
import easyocr

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'geo-clip')))
from geoclip.model.GeoCLIP import GeoCLIP

st.set_page_config(page_title="GeoCLIP AI Locator", page_icon="🌍", layout="centered")

@st.cache_resource
def load_models_and_index():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCLIP().to(device)
    model.eval()
    
    global_index = faiss.read_index("data/global_index.faiss")
    global_cities = pd.read_csv("data/global_cities.csv")
    
    iraq_index = None
    iraq_cities = None
    if os.path.exists("data/iraq_index.faiss"):
        iraq_index = faiss.read_index("data/iraq_index.faiss")
        iraq_cities = pd.read_csv("data/iraq_locations.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return device, model, transform, global_index, global_cities, iraq_index, iraq_cities

@st.cache_resource
def load_ocr_reader():
    # Load EasyOCR for Arabic and English
    return easyocr.Reader(['ar', 'en'], gpu=torch.cuda.is_available())

def get_predictions(image, use_iraq=False, top_k=3):
    device, model, _, global_index, global_cities, iraq_index, iraq_cities = load_models_and_index()
    
    import torch.nn.functional as F
    import torch
    import numpy as np
    
    # 1. TTA (Test-Time Augmentation): 10 Crops
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224)
    ])
    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    crops = base_transform(image)
    img_tensor = torch.stack([to_tensor_norm(crop) for crop in crops]).to(device)
    
    with torch.no_grad():
        img_features_batch = model.image_encoder(img_tensor)
        img_feature = img_features_batch.mean(dim=0, keepdim=True)
        img_feature = F.normalize(img_feature, dim=-1)
        img_features_np = img_feature.cpu().numpy()
        
    if use_iraq and iraq_index is not None:
        index = iraq_index
        if hasattr(index, 'nprobe'):
            index.nprobe = min(16, getattr(index, 'nlist', 16))
        cities = iraq_cities
    else:
        index = global_index
        if hasattr(index, 'nprobe'):
            index.nprobe = min(64, getattr(index, 'nlist', 64))
        cities = global_cities
        
    # Coarse Search
    distances, indices = index.search(img_features_np, top_k)
    
    coarse_candidates = []
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        row = cities.iloc[idx]
        name = row.get('City', 'Unknown')
        if pd.isna(name): name = "Grid Location"
        country = row.get('CountryCode', 'Unknown')
        lat = row['LAT']
        lon = row['LON']
        coarse_candidates.append((lat, lon, name, country, dist))

    # Coarse-to-Fine Micro-Grid
    results = []
    grid_steps = 40
    offset_range = 0.5
    offsets = np.linspace(-offset_range, offset_range, grid_steps)

    for rank, (coarse_lat, coarse_lon, name, country, dist) in enumerate(coarse_candidates):
        lat_grid, lon_grid = np.meshgrid(coarse_lat + offsets, coarse_lon + offsets)
        micro_grid = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
        micro_grid_tensor = torch.tensor(micro_grid, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            loc_features = model.location_encoder(micro_grid_tensor)
            loc_features = F.normalize(loc_features, dim=-1)
            
            logit_scale = model.logit_scale.exp()
            similarity = logit_scale * (img_feature @ loc_features.T)
            
            best_local_idx = similarity.argmax().item()
            best_local_score = similarity[0, best_local_idx].item()
            best_local_coord = micro_grid[best_local_idx]
            
            # Map score roughly to confidence
            raw_cosine = best_local_score / logit_scale.item()
            conf = max(0.0, raw_cosine * 100.0)
            
            results.append({
                "name": name,
                "country": country,
                "lat": best_local_coord[0],
                "lon": best_local_coord[1],
                "distance": best_local_score,
                "confidence": f"{conf:.1f}%"
            })
            
    # Sort results by the new refined score (highest first)
    results.sort(key=lambda x: x['distance'], reverse=True)
    
    return results[:top_k]

def get_exif_location(image):
    try:
        exif = image._getexif()
        if not exif:
            return None
            
        geotagging = {}
        for (idx, tag) in ExifTags.TAGS.items():
            if tag == 'GPSInfo':
                if idx not in exif:
                    return None
                for (key, val) in ExifTags.GPSTAGS.items():
                    if key in exif[idx]:
                        geotagging[val] = exif[idx][key]
                break
                
        if not geotagging or 'GPSLatitude' not in geotagging or 'GPSLongitude' not in geotagging:
            return None

        def get_decimal_from_dms(dms, ref):
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal

        lat = get_decimal_from_dms(geotagging['GPSLatitude'], geotagging['GPSLatitudeRef'])
        lon = get_decimal_from_dms(geotagging['GPSLongitude'], geotagging['GPSLongitudeRef'])
        return lat, lon
    except Exception as e:
        return None

def main():
    st.title("🌍 AI Geo-Locator")
    st.markdown("Upload a photo and the AI will predict its geographic location based on visual features.")
    
    with st.spinner("Initializing AI Core & OCR..."):
        device, _, _, _, _, iraq_idx, _ = load_models_and_index()
        ocr_reader = load_ocr_reader()
        
    st.sidebar.markdown("### Model Settings")
    st.sidebar.markdown(f"**Hardware:** `{device}`")
    
    use_iraq = False
    if iraq_idx is not None:
        use_iraq = st.sidebar.checkbox("🇮🇶 Use Iraq-Only Database", value=True, help="Force the AI to only search within Iraqi borders.")
        
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("📍 Predict Location", type="primary"):
            # Check for EXIF data first
            exif_loc = get_exif_location(image)
            
            if exif_loc:
                st.success("✅ Exact Location Found from Image Metadata (EXIF)!")
                st.markdown(f"""
                <div style="background-color: #2e7d32; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #4CAF50;">
                    <h4 style="margin:0; color: #fff;">🎯 100% Accurate (GPS Metadata)</h4>
                    <p style="margin: 5px 0; font-size: 14px; color: #eee;">
                        <b>Latitude:</b> {exif_loc[0]:.6f} <br>
                        <b>Longitude:</b> {exif_loc[1]:.6f} <br>
                        <b>Confidence Score:</b> 100%
                    </p>
                    <a href="https://www.google.com/maps?q={exif_loc[0]},{exif_loc[1]}" target="_blank" style="text-decoration: none;">
                        <button style="background-color: #fff; color: #2e7d32; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; margin-top: 10px; font-weight: bold;">
                            🗺️ View Exact Location on Google Maps
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
                st.info("The predictions below are the AI's guesses based on visual features, shown for comparison.")
            
            # --- OCR Extraction ---
            with st.spinner("Extracting text from image (OCR)..."):
                img_np = np.array(image)
                ocr_results = ocr_reader.readtext(img_np)
                detected_text = [res[1] for res in ocr_results if res[2] > 0.35]
                
            if detected_text:
                st.markdown(f"""
                <div style="background-color: #2b2b2b; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #4DA8DA;">
                    <h4 style="margin:0; color: #4DA8DA;">📝 Text Detected (OCR Auxiliary Cue)</h4>
                    <p style="margin: 5px 0; font-size: 14px; color: #eee;">
                        The AI read the following text from the image which might help identify the location:<br>
                        <b>{', '.join(detected_text)}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with st.spinner("Analyzing visual features and searching database..."):
                results = get_predictions(image, use_iraq=use_iraq)
                
            if not exif_loc:
                st.success("Analysis Complete!")
                
            st.subheader("Top AI Predictions (Visual Features)")
            for i, res in enumerate(results):
                country_label = res['country'] if not pd.isna(res['country']) else 'GRID'
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #333;">
                        <h4 style="margin:0; color: #4DA8DA;">#{i+1}: {res['name']}, {country_label}</h4>
                        <p style="margin: 5px 0; font-size: 14px; color: #ccc;">
                            <b>Latitude:</b> {res['lat']:.4f} <br>
                            <b>Longitude:</b> {res['lon']:.4f} <br>
                            <b>Confidence Score:</b> {res['confidence']}
                        </p>
                        <a href="https://www.google.com/maps?q={res['lat']},{res['lon']}" target="_blank" style="text-decoration: none;">
                            <button style="background-color: #4CAF50; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; margin-top: 10px;">
                                🗺️ View on Google Maps
                            </button>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
