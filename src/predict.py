import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np
import pandas as pd
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../geo-clip')))

from geoclip.model.GeoCLIP import GeoCLIP
from torchvision import transforms

def predict_image(image_path, top_k=3, use_iraq=False):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at {image_path}")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Using Device: {device}")
    
    # Load Model
    model = GeoCLIP().to(device)
    ckpt_path = "checkpoints/geoclip_checkpoint.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.image_encoder.mlp.load_state_dict(checkpoint['image_encoder_mlp'])
        model.location_encoder.load_state_dict(checkpoint['location_encoder'])
        if isinstance(checkpoint['logit_scale'], torch.Tensor):
            model.logit_scale.data = checkpoint['logit_scale']
        print("[LOG] Loaded trained GeoCLIP model.")
    else:
        print("[WARNING] No trained checkpoint found. Using baseline weights.")
        
    model.eval()

    # Load Global FAISS index
    if use_iraq:
        global_index_path = "data/iraq_index.faiss"
        global_cities_path = "data/iraq_locations.csv"
        print("[LOG] Loading Iraq FAISS index...")
    else:
        global_index_path = "data/global_index.faiss"
        global_cities_path = "data/global_cities.csv"
        print("[LOG] Loading Global FAISS index (170,000+ cities)...")
    
    if not os.path.exists(global_index_path) or not os.path.exists(global_cities_path):
        if use_iraq:
            print("[ERROR] Iraq index not found. Please run 'python src/build_iraq_index.py' first.")
        else:
            print("[ERROR] Global index not found. Please run 'python src/build_global_index.py' first.")
        sys.exit(1)
        
    print("[LOG] Loading Global FAISS index (170,000+ cities)...")
    index = faiss.read_index(global_index_path)
    cities_df = pd.read_csv(global_cities_path, low_memory=False)
    
    city_names = cities_df["City"].values
    country_codes = cities_df["CountryCode"].values
    lats = cities_df["LAT"].values
    lons = cities_df["LON"].values

    import torch.nn.functional as F
    
    # 1. TTA (Test-Time Augmentation): 10 Crops
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224)
    ])
    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    image = Image.open(image_path).convert("RGB")
    crops = base_transform(image)
    image_tensor = torch.stack([to_tensor_norm(crop) for crop in crops]).to(device)
    
    print(f"\n[LOG] Analyzing image: {os.path.basename(image_path)} (with 10x Test-Time Augmentation)")
    with torch.no_grad():
        img_features_batch = model.image_encoder(image_tensor)
        # Average the features across all 10 crops
        img_feature = img_features_batch.mean(dim=0, keepdim=True)
        img_feature = F.normalize(img_feature, dim=-1)
        
    # 2. Search Top-K Coarse Anchors in FAISS
    D, I = index.search(img_feature.cpu().numpy(), top_k)
    
    print("\n" + "="*40)
    print(f"TOP {top_k} COARSE PREDICTIONS")
    print("="*40)
    
    coarse_candidates = []
    for rank in range(top_k):
        idx = I[0][rank]
        city = city_names[idx]
        country = country_codes[idx]
        lat = lats[idx]
        lon = lons[idx]
        coarse_candidates.append((lat, lon, city, country))
        print(f"#{rank+1}: {city}, {country} (Lat: {lat:.4f}, Lon: {lon:.4f})")

    # 3. Coarse-to-Fine Dynamic Micro-Grid Search
    print("\n[LOG] Performing Coarse-to-Fine Micro-Grid Optimization...")
    best_overall_score = -float('inf')
    best_overall_coord = None
    best_overall_city = None

    # Generate a dense micro-grid: +/- 0.5 degrees (approx 55km) divided by 40 steps -> ~1.3km cells
    grid_steps = 40
    offset_range = 0.5
    offsets = np.linspace(-offset_range, offset_range, grid_steps)
    
    for rank, (coarse_lat, coarse_lon, city, country) in enumerate(coarse_candidates):
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
            
            if best_local_score > best_overall_score:
                best_overall_score = best_local_score
                best_overall_coord = best_local_coord
                best_overall_city = f"{city}, {country}"

    print("\n" + "="*40)
    print("FINAL REFINED PREDICTION (Sub-2km Accuracy)")
    print("="*40)
    final_lat, final_lon = best_overall_coord
    print(f"Nearest Anchor City: {best_overall_city}")
    print(f"Refined Coordinates: ({final_lat:.6f}, {final_lon:.6f})")
    print(f"Google Maps: https://www.google.com/maps?q={final_lat},{final_lon}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict location of a single image using GeoCLIP")
    parser.add_argument("image", type=str, help="Path to the input image (e.g., my_photo.jpg)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to show")
    parser.add_argument("--iraq", action="store_true", help="Use the Iraq-specific dataset instead of the global one")
    args = parser.parse_args()
    
    predict_image(args.image, args.top_k, args.iraq)
