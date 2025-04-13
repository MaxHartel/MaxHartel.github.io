import os
from PIL import Image
import pytesseract

# Optional: specify Tesseract path if it's not in your system PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Directory containing images
image_folder = '/Users/maxhartel/Desktop/CsProjects/Website/HandicapperAccuracy/Images'  # change to your folder path
output_file = 'ocr_output.txt'

# Supported image extensions
image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

# Open output file
with open(output_file, 'w', encoding='utf-8') as out:
    for filename in sorted(os.listdir(image_folder)):
        file_path = os.path.join(image_folder, filename)
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in image_extensions:
            try:
                # Open and OCR the image
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img)

                # Write header and text
                out.write(f"=== {filename} ===\n")
                out.write(text.strip() + "\n\n")
                print(f"OCR complete for: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

print(f"\n✅ All text written to {output_file}")
