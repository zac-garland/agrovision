#!/usr/bin/env python3
"""
Automated screenshot tool for AgroVision+ platform.
Takes systematic screenshots of different sections/tabs with a sample image.
"""

import time
import os
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import argparse

# Configuration
DEFAULT_STREAMLIT_URL = "http://localhost:8501"
DEFAULT_SAMPLE_IMAGE = "sample_images/test_plant.jpg"  # Update with your sample image path
BASE_SCREENSHOT_DIR = Path(__file__).parent.parent / "docs" / "screenshots"
BASE_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


class PlatformScreenshotter:
    """Automated screenshot tool for AgroVision platform."""
    
    def __init__(self, streamlit_url=DEFAULT_STREAMLIT_URL, headless=True):
        """
        Initialize screenshot tool.
        
        Args:
            streamlit_url: URL of the Streamlit app
            headless: Run browser in headless mode
        """
        self.streamlit_url = streamlit_url
        self.driver = None
        self.headless = headless
        self.screenshot_dir = None  # Will be set based on input image name
        
    def setup_driver(self):
        """Set up Chrome WebDriver."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,3000")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        
        # Try to use chromedriver from PATH, or specify path if needed
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"❌ Error setting up Chrome driver: {e}")
            print("   Make sure chromedriver is installed and in PATH")
            print("   Install with: brew install chromedriver (on Mac)")
            raise
        
        self.driver.implicitly_wait(10)
        
    def wait_for_element(self, by, value, timeout=30):
        """Wait for element to be present."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            print(f"⚠️  Timeout waiting for element: {by}={value}")
            return None
    
    def upload_image(self, image_path):
        """
        Upload an image to the Streamlit file uploader.
        
        Args:
            image_path: Path to the image file to upload
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Sample image not found: {image_path}")
        
        # Find the file uploader
        try:
            # Streamlit file uploader is typically a file input
            file_input = self.driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
            file_input.send_keys(str(image_path.absolute()))
            print(f"✅ Uploaded image: {image_path.name}")
            
            # Wait for processing (adjust timeout as needed)
            time.sleep(20)  # Wait for backend processing
            
        except NoSuchElementException:
            print("⚠️  File uploader not found. Trying alternative method...")
            # Alternative: use JavaScript to trigger file upload
            self.driver.execute_script("""
                const input = document.querySelector('input[type="file"]');
                if (input) {
                    const file = new File([''], arguments[0]);
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    input.files = dataTransfer.files;
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            """, str(image_path.absolute()))
            time.sleep(20)
    
    def _clean_filename(self, filename):
        """
        Clean filename to create a valid directory name.
        Removes extension, replaces spaces with underscores, removes special chars.
        
        Args:
            filename: Original filename (e.g., "test 1.jpeg")
            
        Returns:
            Cleaned name (e.g., "test_1")
        """
        # Get base name without extension
        name = Path(filename).stem
        
        # Replace spaces with underscores
        name = name.replace(' ', '_')
        
        # Remove or replace special characters (keep alphanumeric, underscore, hyphen)
        import re
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        
        # Remove multiple consecutive underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        return name if name else "screenshot"
    
    def _setup_screenshot_dir(self, image_path):
        """
        Set up screenshot directory based on input image name.
        
        Args:
            image_path: Path to the input image file
        """
        clean_name = self._clean_filename(image_path)
        self.screenshot_dir = BASE_SCREENSHOT_DIR / clean_name
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Screenshot directory: {self.screenshot_dir}")
    
    def _get_full_page_dimensions(self):
        """
        Get the full page dimensions including scrollable content.
        
        Returns:
            Tuple of (width, height) for the full page
        """
        # Get viewport dimensions
        viewport_width = self.driver.execute_script("return window.innerWidth")
        viewport_height = self.driver.execute_script("return window.innerHeight")
        
        # Get full page dimensions (including scrollable content)
        full_width = self.driver.execute_script("return Math.max(document.body.scrollWidth, document.body.offsetWidth, document.documentElement.clientWidth, document.documentElement.scrollWidth, document.documentElement.offsetWidth)")
        full_height = self.driver.execute_script("return Math.max(document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight)")
        
        return full_width, full_height
    
    def take_screenshot(self, filename, element=None, full_page=True):
        """
        Take a screenshot.
        
        Args:
            filename: Name for the screenshot file
            element: Optional element to screenshot (if None, screenshots entire page)
            full_page: If True, capture the full page height (default: True)
        """
        if self.screenshot_dir is None:
            # Fallback to base directory if not set
            filepath = BASE_SCREENSHOT_DIR / filename
        else:
            filepath = self.screenshot_dir / filename
        
        if element:
            element.screenshot(str(filepath))
        else:
            if full_page:
                # Get full page dimensions
                full_width, full_height = self._get_full_page_dimensions()
                
                # Chrome has a maximum window size limit (typically 32767x32767)
                # But we'll use a more reasonable limit
                max_height = 32767
                max_width = 32767
                
                # Get current window size
                current_size = self.driver.get_window_size()
                current_width = current_size['width']
                current_height = current_size['height']
                
                # Resize window to fit full page (within limits)
                new_width = min(full_width, max_width)
                new_height = min(full_height, max_height)
                
                # Only resize if needed
                if new_width != current_width or new_height != current_height:
                    self.driver.set_window_size(new_width, new_height)
                    time.sleep(1)  # Wait for resize to complete
                
                # Scroll to top to ensure we capture from the beginning
                self.driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(0.5)
            
            # Take screenshot
            self.driver.save_screenshot(str(filepath))
            
            # Restore original window size if we changed it
            if full_page and (new_width != current_width or new_height != current_height):
                self.driver.set_window_size(current_width, current_height)
        
        print(f"📸 Saved screenshot: {filepath}")
        return filepath
    
    def navigate_to_tab(self, tab_name_or_index):
        """
        Navigate to a specific tab in Streamlit using HTML position or name.
        
        Args:
            tab_name_or_index: Either tab name (e.g., "Diagnosis") or index (0, 1, 2, 3)
        """
        try:
            # Find all tabs using Streamlit's data-testid
            tabs = self.driver.find_elements(By.CSS_SELECTOR, 'button[data-testid="stTab"]')
            
            if not tabs:
                print("⚠️  No tabs found with data-testid='stTab'")
                return False
            
            # If it's a number, use index
            if isinstance(tab_name_or_index, int) or str(tab_name_or_index).isdigit():
                tab_index = int(tab_name_or_index)
                if 0 <= tab_index < len(tabs):
                    tabs[tab_index].click()
                    time.sleep(2)  # Wait for tab to load
                    print(f"✅ Navigated to tab at index {tab_index}")
                    return True
                else:
                    print(f"⚠️  Tab index {tab_index} out of range (0-{len(tabs)-1})")
                    return False
            
            # Otherwise, search by text content
            tab_name = str(tab_name_or_index)
            for i, tab in enumerate(tabs):
                try:
                    # Get text from the tab (may include emoji)
                    tab_text = tab.text.strip()
                    # Check if tab name matches (case-insensitive, partial match)
                    if tab_name.lower() in tab_text.lower() or tab_text.lower() in tab_name.lower():
                        tab.click()
                        time.sleep(2)  # Wait for tab to load
                        print(f"✅ Navigated to tab: {tab_text} (index {i})")
                        return True
                except Exception as e:
                    continue
            
            print(f"⚠️  Tab '{tab_name}' not found. Available tabs: {[t.text for t in tabs]}")
            return False
            
        except Exception as e:
            print(f"⚠️  Error navigating to tab '{tab_name_or_index}': {e}")
            return False
    
    def navigate_to_tab_by_index(self, index):
        """
        Navigate to tab by its position in the tab list (0-based).
        
        Args:
            index: Tab index (0 = Diagnosis, 1 = Species, 2 = Model Results, 3 = Leaf Analysis)
        """
        return self.navigate_to_tab(index)
    
    def capture_all_sections(self, sample_image_path):
        """
        Capture screenshots of all sections/tabs.
        
        Args:
            sample_image_path: Path to sample image to upload
        """
        # Set up screenshot directory based on image name
        self._setup_screenshot_dir(sample_image_path)
        
        print(f"🚀 Starting screenshot capture...")
        print(f"   Streamlit URL: {self.streamlit_url}")
        print(f"   Sample image: {sample_image_path}")
        print(f"   Screenshot dir: {self.screenshot_dir}")
        print()
        
        # Navigate to Streamlit app
        print("📱 Navigating to Streamlit app...")
        self.driver.get(self.streamlit_url)
        time.sleep(5)  # Wait for page to load
        
        # Take initial screenshot (landing page)
        self.take_screenshot("01_landing_page.png")
        
        # Upload sample image
        print("\n📤 Uploading sample image...")
        self.upload_image(sample_image_path)
        time.sleep(5)  # Wait for processing
        
        # Take screenshot after upload (before results)
        self.take_screenshot("02_after_upload.png")
        
        # Wait for diagnosis to complete
        print("\n⏳ Waiting for diagnosis to complete...")
        time.sleep(30)  # Adjust based on your backend processing time
        
        # Capture each tab/section by index (more reliable than name matching)
        # Tab order based on HTML: 0=Diagnosis, 1=Species, 2=Model Results, 3=Leaf Analysis
        tab_info = [
            (0, "diagnosis"),
            (1, "species"),
            (2, "model_results"),
            (3, "leaf_analysis")
        ]
        
        for i, (tab_index, tab_name) in enumerate(tab_info, start=3):
            print(f"\n📑 Capturing tab {tab_index}: {tab_name}")
            if self.navigate_to_tab_by_index(tab_index):
                time.sleep(4)  # Wait for content to load
                self.take_screenshot(f"{i:02d}_tab_{tab_name}.png")
            else:
                # Fallback: try by name
                tab_display_names = ["Diagnosis", "Species", "Model Results", "Leaf Analysis"]
                if self.navigate_to_tab(tab_display_names[tab_index]):
                    time.sleep(4)
                    self.take_screenshot(f"{i:02d}_tab_{tab_name}.png")
                else:
                    # Last resort: take screenshot anyway
                    time.sleep(2)
                    self.take_screenshot(f"{i:02d}_tab_{tab_name}_attempt.png")
        
        # Take final full-page screenshot
        print("\n📸 Taking final full-page screenshot...")
        self.driver.execute_script("window.scrollTo(0, 0);")  # Scroll to top
        time.sleep(5)
        self.take_screenshot("99_full_page.png")
        
        print(f"\n✅ Screenshot capture complete!")
        print(f"   Screenshots saved to: {self.screenshot_dir}")
    
    def close(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()
            print("🔒 Browser closed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Take screenshots of AgroVision+ platform")
    parser.add_argument(
        "--url",
        default=DEFAULT_STREAMLIT_URL,
        help=f"Streamlit app URL (default: {DEFAULT_STREAMLIT_URL})"
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_SAMPLE_IMAGE,
        help=f"Path to sample image (default: {DEFAULT_SAMPLE_IMAGE})"
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in visible mode (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Check if sample image exists
    if not Path(args.image).exists():
        print(f"⚠️  Sample image not found: {args.image}")
        print("   Please provide a valid image path with --image")
        return
    
    # Create screenshotter
    screenshotter = PlatformScreenshotter(
        streamlit_url=args.url,
        headless=not args.no_headless
    )
    
    try:
        screenshotter.setup_driver()
        screenshotter.capture_all_sections(args.image)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        screenshotter.close()


if __name__ == "__main__":
    main()

