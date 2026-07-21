
import os
from PIL import Image
from pdf2image import convert_from_path

# --- Configuration ---
INPUT_DIR = "raw_mixed_data"
OUTPUT_DIR = "dataset/raw/authentic"

def standardize_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success, errors = 0, 0
    
    print(f"Scanning '{INPUT_DIR}'...")
    
    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)
        ext = filename.lower().split('.')[-1]
        
        try:
            # 1. Convert PDFs to JPG
            if ext == 'pdf':
                # Convert the first page of the PDF to an image
                images = convert_from_path(file_path, first_page=1, last_page=1)
                if images:
                    img = images[0].convert('RGB')
                    new_filename = filename.replace('.pdf', '.jpg').replace('.PDF', '.jpg')
                    save_path = os.path.join(OUTPUT_DIR, new_filename)
                    # Save as high-quality JPG
                    img.save(save_path, 'JPEG', quality=95)
                    success += 1
                    print(f"Converted PDF: {filename} -> {new_filename}")
                    
            # 2. Copy existing JPGs/PNGs over and ensure standard format
            elif ext in ['jpg', 'jpeg', 'png']:
                img = Image.open(file_path).convert('RGB')
                new_filename = filename.rsplit('.', 1)[0] + '.jpg'
                save_path = os.path.join(OUTPUT_DIR, new_filename)
                img.save(save_path, 'JPEG', quality=95)
                success += 1
                print(f"Standardized Image: {filename} -> {new_filename}")
                
        except Exception as e:
            print(f"Error on {filename}: {e}")
            errors += 1

    print(f"\nDone! Successfully saved {success} JPGs to '{OUTPUT_DIR}'.")
    if errors > 0:
        print(f"Encountered {errors} errors. Check the logs above.")

if __name__ == "__main__":
    standardize_data()