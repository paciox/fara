"""Desktop controller using PyAutoGUI for computer automation."""

import pyautogui
from PIL import Image
import time
import logging
from typing import Tuple, List
import io


class DesktopController:
    """Controller for desktop automation using PyAutoGUI."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize desktop controller with safety settings."""
        self.logger = logger or logging.getLogger(__name__)
        
        # Safety: Move mouse to corner to abort
        pyautogui.FAILSAFE = True
        
        # Pause between actions for stability
        pyautogui.PAUSE = 0.3
        
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        self.logger.info(
            f"Desktop controller initialized. Screen size: {self.screen_width}x{self.screen_height}"
        )

    async def get_screenshot(self, path: str | None = None) -> bytes:
        """
        Capture screenshot of the entire desktop.
        
        Args:
            path: Optional path to save screenshot to disk
            
        Returns:
            Screenshot as bytes
        """
        screenshot = pyautogui.screenshot()
        
        if path:
            screenshot.save(path)
            self.logger.debug(f"Screenshot saved to {path}")
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    async def click_coords(
        self, x: int, y: int, button: str = "left", clicks: int = 1, duration: float = 0.5
    ) -> None:
        """
        Click at specific coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button ('left', 'right', 'middle')
            clicks: Number of clicks (1=single, 2=double)
            duration: Duration of mouse movement in seconds
        """
        self.logger.debug(f"Clicking at ({x}, {y}) with {button} button")
        
        # Move mouse smoothly to coordinates
        pyautogui.moveTo(x, y, duration=duration)
        
        # Small delay before click
        time.sleep(0.1)
        
        # Click
        pyautogui.click(x, y, clicks=clicks, button=button)
        
        # Wait for action to complete
        time.sleep(0.5)

    async def type_text(self, text: str, interval: float = 0.05) -> None:
        """
        Type text at current cursor position.
        
        Args:
            text: Text to type
            interval: Interval between keystrokes
        """
        self.logger.debug(f"Typing text: {text[:50]}...")
        pyautogui.write(text, interval=interval)
        time.sleep(0.3)

    async def press_keys(self, keys: List[str]) -> None:
        """
        Press keyboard keys (supports hotkeys).
        
        Args:
            keys: List of keys to press (e.g., ['ctrl', 'c'])
        """
        self.logger.debug(f"Pressing keys: {keys}")
        
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            # Hotkey combination
            pyautogui.hotkey(*keys)
        
        time.sleep(0.3)

    async def scroll(self, clicks: int) -> None:
        """
        Scroll vertically.
        
        Args:
            clicks: Positive = scroll up, Negative = scroll down
        """
        self.logger.debug(f"Scrolling {clicks} clicks")
        pyautogui.scroll(clicks)
        time.sleep(0.3)

    async def move_mouse(self, x: int, y: int, duration: float = 0.5) -> None:
        """
        Move mouse to coordinates without clicking.
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration: Duration of movement
        """
        self.logger.debug(f"Moving mouse to ({x}, {y})")
        pyautogui.moveTo(x, y, duration=duration)
        time.sleep(0.2)

    async def drag(self, x: int, y: int, duration: float = 1.0) -> None:
        """
        Drag from current position to target coordinates.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Duration of drag
        """
        self.logger.debug(f"Dragging to ({x}, {y})")
        pyautogui.dragTo(x, y, duration=duration, button='left')
        time.sleep(0.3)

    async def sleep(self, duration: float) -> None:
        """
        Wait/sleep for a duration.
        
        Args:
            duration: Sleep duration in seconds
        """
        self.logger.debug(f"Sleeping for {duration} seconds")
        time.sleep(duration)

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions."""
        return self.screen_width, self.screen_height
