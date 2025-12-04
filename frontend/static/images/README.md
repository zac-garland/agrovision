# Banner Image Directory

Place your AgroVision banner image in this directory.

## Supported File Names

The banner function will automatically look for images with these names (in order):
1. `banner.png`
2. `banner.jpg`
3. `agrovision_banner.png`
4. `agrovision_banner.jpg`

## Recommended Specifications

- **Format**: PNG or JPG
- **Dimensions**: 1200-1600px wide (will be automatically resized)
- **Aspect Ratio**: Approximately 3:1 to 4:1 (width:height)
- **Content**: Your banner image with the robot examining plants

## How It Works

The banner function will:
1. Load the image from this directory
2. Resize it to fit the page width (max 1200px)
3. Add a semi-transparent green overlay for better text readability
4. Display the image with the AgroVision+ title and subtitle overlaid on top

## Instructions

1. Save your banner image to this directory with one of the supported filenames
2. The image will automatically appear as the background when you run the Streamlit app
3. If no image is found, the app will fall back to a styled gradient banner

