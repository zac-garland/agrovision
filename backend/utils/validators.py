from PIL import Image
import io
from config import ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE_MB

def validate_image_file(file_obj):
    """
    Validate uploaded image file.
    
    Args:
        file_obj: werkzeug FileStorage object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check file exists
    if not file_obj or file_obj.filename == '':
        return False, "No file selected"
    
    # Check extension
    ext = file_obj.filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check size
    file_obj.seek(0, 2)  # Seek to end
    file_size_mb = file_obj.tell() / (1024 * 1024)
    if file_size_mb > MAX_IMAGE_SIZE_MB:
        return False, f"File too large. Max: {MAX_IMAGE_SIZE_MB}MB"
    
    file_obj.seek(0)  # Reset to start
    
    # Try to open as image
    try:
        img = Image.open(file_obj)
        img.verify()
        file_obj.seek(0)  # Reset after verify
        
        # Reopen for actual use (verify closes it)
        img = Image.open(file_obj)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        return True, img
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"