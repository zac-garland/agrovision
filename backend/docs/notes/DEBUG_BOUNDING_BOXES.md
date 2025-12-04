# Debugging Bounding Box Visualization

## What Was Changed

1. **Always show annotated image** - Even if no boxes found
2. **Improved box drawing** - Brighter colors (lime green), thicker lines (5px)
3. **Better box extraction** - Gets boxes from both `leaf_boxes` array and `individual_leaves`
4. **Debug output** - Expandable section shows what data was received

## How to Debug

1. **Check the Debug expander** in the frontend:
   - Look for "🔍 Debug: Bounding Box Info"
   - See what boxes are in the response

2. **Check browser console** for errors

3. **Check backend logs** for:
   - "Running leaf detection..."
   - Number of leaves detected
   - Any errors

## Common Issues

### Boxes not showing
- **Check**: Are boxes in the response? (Look at debug expander)
- **Check**: Are coordinates valid? (Should be numbers, not None)
- **Check**: Image size matches coordinates?

### Boxes are outside image
- Coordinates might be wrong format
- Image might be resized/cropped

### No boxes in response
- YOLO might not be detecting leaves
- Check backend logs for detection results
- Verify YOLO model file exists

## Next Steps

1. Upload an image and run diagnosis
2. Check the debug expander to see what boxes were received
3. If boxes are empty, check backend logs for YOLO detection errors
4. If boxes exist but don't show, check coordinates format

