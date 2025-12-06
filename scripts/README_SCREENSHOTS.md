# Platform Screenshot Automation

Automated tool for taking systematic screenshots of the AgroVision+ platform.

## Setup

1. Install dependencies:
```bash
pip install selenium webdriver-manager
```

2. Install ChromeDriver:
```bash
# On Mac:
brew install chromedriver

# On Linux:
# Download from https://chromedriver.chromium.org/
# Or use webdriver-manager (automatic)
```

## Usage

### Basic Usage

```bash
# Make sure Streamlit app is running on localhost:8501
# Then run:
python scripts/screenshot_platform.py --image path/to/sample/image.jpg
```

### Options

```bash
# Specify custom Streamlit URL
python scripts/screenshot_platform.py --url http://localhost:8502 --image sample.jpg

# Run in visible mode (for debugging)
python scripts/screenshot_platform.py --no-headless --image sample.jpg

# Use different sample image
python scripts/screenshot_platform.py --image test_images/begonia.jpg
```

## What It Does

1. Opens the Streamlit app in a browser
2. Takes a screenshot of the landing page
3. Uploads a sample image
4. Waits for diagnosis to complete
5. Captures screenshots of each tab:
   - Diagnosis
   - Species
   - Model Results
   - Leaf Analysis
6. Takes a final full-page screenshot

## Output

Screenshots are saved to `docs/screenshots/` with descriptive names:
- `01_landing_page.png`
- `02_after_upload.png`
- `03_tab_diagnosis.png`
- `04_tab_species.png`
- `05_tab_model_results.png`
- `06_tab_leaf_analysis.png`
- `99_full_page.png`

## Troubleshooting

### ChromeDriver not found
- Install ChromeDriver: `brew install chromedriver` (Mac)
- Or use webdriver-manager for automatic management

### File upload not working
- Make sure the image path is absolute
- Check that Streamlit file uploader is visible
- Try running in non-headless mode to debug

### Tabs not found
- Streamlit tab structure may have changed
- Check the actual HTML structure in browser dev tools
- Update selectors in `navigate_to_tab()` method

### Timeout errors
- Increase wait times in the script
- Make sure backend is running and responsive
- Check network connectivity

