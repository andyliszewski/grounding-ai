"""
Create a simple test formula image for pix2tex PoC
"""
from PIL import Image, ImageDraw, ImageFont
import os

# Create a simple white image with a formula text
width, height = 400, 100
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# Try to use a system font, fallback to default if not available
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
except:
    font = ImageFont.load_default()

# Draw a simple formula text (not actual LaTeX rendering, just text)
# This simulates what a scanned formula might look like
formula_text = "E = mc²"
bbox = draw.textbbox((0, 0), formula_text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (width - text_width) // 2
y = (height - text_height) // 2
draw.text((x, y), formula_text, fill='black', font=font)

# Save the image
output_path = os.path.join(os.path.dirname(__file__), 'formula_test_simple.png')
img.save(output_path)
print(f"Created test formula image: {output_path}")
