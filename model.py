from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Load your trained YOLO model
model = YOLO('brain_model.pt')

classes = ['Glioma', 'Meningioma', 'Pituitary tumor']
def predict_tumor(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Perform prediction
    results = model(image)
    
    # Process the results
    result = results[0]  # Get the first (and only) result
    
    if len(result.boxes) == 0:
        return "No tumor detected", 0.0, image, Image.new('L', image.size)
    
    # Get the box with the highest confidence
    best_box = max(result.boxes, key=lambda box: box.conf.item())
    
    class_id = int(best_box.cls.item())
    confidence = best_box.conf.item()
    
    predicted_class = classes[class_id]
    
    # Get the bounding box coordinates
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    
    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Modern style: gradient-filled rectangle
    gradient = create_gradient((x2-x1, y2-y1), (255, 255, 0, 64), (255, 0, 0, 64))
    draw_image.paste(gradient, (x1, y1), gradient)
    
    # Dashed border
    draw_dashed_rectangle(draw, [x1, y1, x2, y2], (255, 255, 0), width=2, dash_length=10)
    
    # Add label
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    label = f"{predicted_class}: {confidence:.2f}"
    
    # Get text bounding box
    left, top, right, bottom = draw.textbbox((x1, y1-25), label, font=font)
    
    # Draw text background with rounded corners
    draw_rounded_rectangle(draw, [left-5, top-5, right+5, bottom+5], 10, fill=(255, 255, 0, 200))
    
    # Draw text
    draw.text((x1, y1-25), label, fill=(0, 0, 0), font=font)
    
    # Create segmentation mask
    segmentation_mask = create_segmentation_mask(image, x1, y1, x2, y2)
    
    return predicted_class, confidence, draw_image, segmentation_mask

def create_gradient(size, color1, color2):
    base = Image.new('RGBA', size, color1)
    top = Image.new('RGBA', size, color2)
    mask = Image.new('L', size)
    mask_data = []
    for y in range(size[1]):
        mask_data.extend([int(255 * (y / size[1]))] * size[0])
    mask.putdata(mask_data)
    return Image.composite(base, top, mask)

def draw_dashed_rectangle(draw, xy, color, width=1, dash_length=10):
    x1, y1, x2, y2 = xy
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        dash_count = int(length / (2 * dash_length))
        for j in range(dash_count):
            s = j / dash_count
            e = (j + 0.5) / dash_count
            s_point = (int(start[0] + (end[0] - start[0]) * s), int(start[1] + (end[1] - start[1]) * s))
            e_point = (int(start[0] + (end[0] - start[0]) * e), int(start[1] + (end[1] - start[1]) * e))
            draw.line([s_point, e_point], fill=color, width=width)

def draw_rounded_rectangle(draw, xy, corner_radius, fill):
    x1, y1, x2, y2 = xy
    draw.rectangle([x1+corner_radius, y1, x2-corner_radius, y2], fill=fill)
    draw.rectangle([x1, y1+corner_radius, x2, y2-corner_radius], fill=fill)
    draw.pieslice([x1, y1, x1+corner_radius*2, y1+corner_radius*2], 180, 270, fill=fill)
    draw.pieslice([x2-corner_radius*2, y1, x2, y1+corner_radius*2], 270, 0, fill=fill)
    draw.pieslice([x1, y2-corner_radius*2, x1+corner_radius*2, y2], 90, 180, fill=fill)
    draw.pieslice([x2-corner_radius*2, y2-corner_radius*2, x2, y2], 0, 90, fill=fill)

def create_segmentation_mask(image, x1, y1, x2, y2):
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create a mask for the bounding box
    mask = np.zeros(cv_image.shape[:2], np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    # Apply GrabCut algorithm
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (x1, y1, x2-x1, y2-y1)
    cv2.grabCut(cv_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create a mask where sure and likely foreground are set to 1, otherwise 0
    mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    # Convert mask back to PIL Image
    return Image.fromarray(mask * 255)

def get_model_info():
    return {
        'type': type(model).__name__,
        'task': model.task,
        'num_classes': len(classes),
        'classes': classes
    }