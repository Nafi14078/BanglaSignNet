import os
import pandas as pd
import json

def create_annotations_file():
    data_path = \"data/raw/BSL-D\"
    annotations = []
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        print(f\"Error: Data path {data_path} does not exist!\")
        print(\"Please download the BSL-D dataset first.\")
        return None
    
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            for video_file in os.listdir(class_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(class_path, video_file)
                    annotations.append({
                        'video_path': video_path,
                        'label': class_name,
                        'label_id': int(class_name.split('_')[0])
                    })
    
    # Create DataFrame
    df = pd.DataFrame(annotations)
    
    # Create annotations directory if it doesn't exist
    os.makedirs('data/annotations', exist_ok=True)
    
    df.to_csv('data/annotations/bsld_annotations.csv', index=False)
    
    # Create label mapping
    unique_labels = df[['label_id', 'label']].drop_duplicates()
    label_to_id = {row['label']: row['label_id'] for _, row in unique_labels.iterrows()}
    id_to_label = {row['label_id']: row['label'] for _, row in unique_labels.iterrows()}
    
    with open('data/annotations/label_mapping.json', 'w') as f:
        json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f, indent=2)
    
    print(f\"Created annotations for {len(df)} videos\")
    print(f\"Number of classes: {len(unique_labels)}\")
    
    return df

if __name__ == \"__main__\":
    create_annotations_file()
