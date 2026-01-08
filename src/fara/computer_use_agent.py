"""Computer Use Agent - Desktop automation with FARA."""

from typing import Any, List, Tuple, Dict
import logging
import json
import ast
import io
import os
from PIL import Image
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from .desktop.desktop_controller import DesktopController
from ._prompts import get_computer_use_system_prompt
from .fara_types import (
    LLMMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ImageObj,
    ModelResponse,
    FunctionCall,
    message_to_openai_format,
)


class ComputerUseAgent:
    """Agent for desktop/computer automation using vision + PyAutoGUI."""

    SCREENSHOT_TOKENS = 1105
    USER_MESSAGE = "Here is the next screenshot. Think about what to do next."
    
    MLM_PROCESSOR_IM_CFG = {
        "min_pixels": 3136,
        "max_pixels": 12845056,
        "patch_size": 14,
        "merge_size": 2,
    }

    def __init__(
        self,
        client_config: dict,
        downloads_folder: str | None = None,
        max_n_images: int = 3,
        fn_call_template: str = "default",
        model_call_timeout: int = 20,
        max_rounds: int = 10,
        save_screenshots: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.client_config = client_config
        self.downloads_folder = downloads_folder
        self.max_n_images = max_n_images
        self.fn_call_template = fn_call_template
        self.model_call_timeout = model_call_timeout
        self.max_rounds = max_rounds
        self.save_screenshots = save_screenshots
        self.logger = logger or logging.getLogger(__name__)
        
        if self.downloads_folder and not os.path.exists(self.downloads_folder):
            os.makedirs(self.downloads_folder)
        
        # Initialize desktop controller
        self.desktop_controller = DesktopController(logger=self.logger)
        
        # Get screen dimensions
        self.viewport_width, self.viewport_height = self.desktop_controller.get_screen_size()
        self.logger.info(f"Screen size: {self.viewport_width}x{self.viewport_height}")
        
        # MLM processing dimensions
        self._mlm_width = 1440
        self._mlm_height = 900
        
        self._num_actions = 0
        self._chat_history: List[LLMMessage] = []
        self._openai_client: AsyncOpenAI | None = None
        self.include_input_text_key_args = True

    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        self._openai_client = AsyncOpenAI(
            api_key=self.client_config.get("api_key"),
            base_url=self.client_config.get("base_url"),
        )
        self.logger.info("Computer Use Agent initialized")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=5.0, min=5.0, max=60),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    async def _make_model_call(
        self,
        history: List[LLMMessage],
        extra_create_args: Dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Make a model call using OpenAI client."""
        openai_messages = [message_to_openai_format(msg) for msg in history]
        request_params = {
            "model": self.client_config.get("model", "gpt-4o"),
            "messages": openai_messages,
        }
        if extra_create_args:
            request_params.update(extra_create_args)

        response = await self._openai_client.chat.completions.create(**request_params)
        content = response.choices[0].message.content
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return ModelResponse(content=content, usage=usage)

    async def _get_scaled_screenshot(self) -> Image.Image:
        """Get current screenshot and scale it for the model."""
        screenshot_bytes = await self.desktop_controller.get_screenshot()
        screenshot = Image.open(io.BytesIO(screenshot_bytes))
        _, scaled_screenshot = self._get_system_message(screenshot)
        return scaled_screenshot

    def _get_system_message(
        self, screenshot: ImageObj | Image.Image
    ) -> Tuple[List[SystemMessage], Image.Image]:
        """Generate system message with screenshot."""
        system_prompt_info = get_computer_use_system_prompt(
            screenshot,
            self.MLM_PROCESSOR_IM_CFG,
            include_input_text_key_args=self.include_input_text_key_args,
            fn_call_template=self.fn_call_template,
        )
        self._mlm_width, self._mlm_height = system_prompt_info["im_size"]
        scaled_screenshot = screenshot.resize((self._mlm_width, self._mlm_height))

        system_message = []
        for msg in system_prompt_info["conversation"]:
            tmp_content = ""
            for content in msg["content"]:
                tmp_content += content["text"]
            system_message.append(SystemMessage(content=tmp_content))

        return system_message, scaled_screenshot

    def _parse_thoughts_and_action(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """Parse LLM response into thoughts and action."""
        try:
            tmp = message.split("<tool_call>\n")
            thoughts = tmp[0].strip()
            action_text = tmp[1].split("\n</tool_call>")[0]
            try:
                action = json.loads(action_text)
            except json.decoder.JSONDecodeError:
                self.logger.error(f"Invalid action text: {action_text}")
                action = ast.literal_eval(action_text)
            return thoughts, action
        except Exception as e:
            self.logger.error(f"Error parsing thoughts and action: {message}", exc_info=True)
            raise e

    def proc_coords(
        self,
        coords: List[float] | None,
        im_w: int,
        im_h: int,
        og_im_w: int | None = None,
        og_im_h: int | None = None,
    ) -> List[float] | None:
        """Scale coordinates from model resolution to actual screen resolution."""
        if not coords:
            return coords

        if og_im_w is None:
            og_im_w = im_w
        if og_im_h is None:
            og_im_h = im_h

        tgt_x, tgt_y = coords
        scale_x = og_im_w / im_w
        scale_y = og_im_h / im_h
        return [tgt_x * scale_x, tgt_y * scale_y]

    def maybe_remove_old_screenshots(
        self, history: List[LLMMessage], includes_current: bool = False
    ) -> List[LLMMessage]:
        """Remove old screenshots to manage context window."""
        if self.max_n_images <= 0:
            return history

        max_n_images = self.max_n_images if includes_current else self.max_n_images - 1
        new_history: List[LLMMessage] = []
        n_images = 0
        
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            
            if isinstance(msg.content, list):
                has_image = any(isinstance(c, ImageObj) for c in msg.content)
                if has_image:
                    if n_images < max_n_images:
                        new_history.append(msg)
                    n_images += 1
                else:
                    new_history.append(msg)
            elif isinstance(msg.content, ImageObj):
                if n_images < max_n_images:
                    new_history.append(msg)
                n_images += 1
            else:
                new_history.append(msg)

        return new_history[::-1]

    async def run(self, user_message: str) -> Tuple:
        """Run the agent with a user message."""
        await self.initialize()

        # Get initial screenshot
        scaled_screenshot = await self._get_scaled_screenshot()

        if self.save_screenshots:
            await self.desktop_controller.get_screenshot(
                path=os.path.join(self.downloads_folder, f"screenshot{self._num_actions}.png")
            )

        self._chat_history.append(
            UserMessage(
                content=[ImageObj.from_pil(scaled_screenshot), user_message],
                is_original=True,
            )
        )

        all_actions = []
        all_observations = []
        final_answer = "<no_answer>"
        is_stop_action = False
        
        for i in range(self.max_rounds):
            is_first_round = i == 0
            
            function_call, raw_response = await self.generate_model_call(
                is_first_round, scaled_screenshot if is_first_round else None
            )
            
            all_actions.append(raw_response)
            thoughts, action_dict = self._parse_thoughts_and_action(raw_response)
            action_args = action_dict.get("arguments", {})
            action = action_args["action"]
            
            self.logger.debug(
                f"\nThought #{i+1}: {thoughts}\nAction #{i+1}: executing '{action}' with {json.dumps(action_args)}"
            )
            print(
                f"\nThought #{i+1}: {thoughts}\nAction #{i+1}: executing '{action}' with {json.dumps(action_args)}"
            )
            
            is_stop_action, action_description = await self.execute_action(function_call)
            all_observations.append(action_description)
            
            self.logger.debug(f"Observation#{i+1}: {action_description}")
            print(f"Observation#{i+1}: {action_description}")
            
            if is_stop_action:
                final_answer = thoughts
                break
        
        return final_answer, all_actions, all_observations

    async def generate_model_call(
        self, is_first_round: bool, first_screenshot: Image.Image | None = None
    ) -> Tuple[List[FunctionCall], str]:
        """Generate model call and get action decision."""
        history = self.maybe_remove_old_screenshots(self._chat_history)

        screenshot_for_system = first_screenshot
        if not is_first_round:
            scaled_screenshot = await self._get_scaled_screenshot()
            screenshot_for_system = scaled_screenshot

            curr_message = UserMessage(
                content=[ImageObj.from_pil(scaled_screenshot), self.USER_MESSAGE]
            )
            self._chat_history.append(curr_message)
            history.append(curr_message)

        system_message, _ = self._get_system_message(screenshot_for_system)
        history = system_message + history
        
        response = await self._make_model_call(
            history, extra_create_args={"temperature": 0}
        )
        message = response.content

        self._chat_history.append(AssistantMessage(content=message))
        thoughts, action = self._parse_thoughts_and_action(message)
        action["arguments"]["thoughts"] = thoughts

        function_call = [FunctionCall(id="dummy", **action)]
        return function_call, message

    async def execute_action(self, function_call: List[FunctionCall]) -> Tuple[bool, str]:
        """Execute the action on desktop."""
        args = function_call[0].arguments
        action_description = ""
        is_stop_action = False

        # Scale coordinates if present
        if "coordinate" in args:
            args["coordinate"] = self.proc_coords(
                args["coordinate"],
                self._mlm_width,
                self._mlm_height,
                self.viewport_width,
                self.viewport_height,
            )

        action = args["action"]

        if action == "click" or action == "left_click":
            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                action_description = f"I clicked at coordinates ({int(tgt_x)}, {int(tgt_y)})."
                await self.desktop_controller.click_coords(int(tgt_x), int(tgt_y))

        elif action == "right_click":
            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                action_description = f"I right-clicked at ({int(tgt_x)}, {int(tgt_y)})."
                await self.desktop_controller.click_coords(int(tgt_x), int(tgt_y), button="right")

        elif action == "double_click":
            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                action_description = f"I double-clicked at ({int(tgt_x)}, {int(tgt_y)})."
                await self.desktop_controller.click_coords(int(tgt_x), int(tgt_y), clicks=2)

        elif action == "input_text" or action == "type":
            text_value = str(args.get("text", args.get("text_value", "")))
            action_description = f"I typed '{text_value}'."
            
            # Click first if coordinate provided
            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                await self.desktop_controller.click_coords(int(tgt_x), int(tgt_y))
            
            await self.desktop_controller.type_text(text_value)

        elif action == "keypress" or action == "key":
            keys = args.get("keys", [])
            action_description = f"I pressed keys: {keys}"
            await self.desktop_controller.press_keys(keys)

        elif action == "scroll":
            pixels = int(args.get("pixels", 0))
            clicks = pixels // 100  # Convert pixels to scroll clicks
            if clicks > 0:
                action_description = f"I scrolled up {abs(clicks)} clicks."
            else:
                action_description = f"I scrolled down {abs(clicks)} clicks."
            await self.desktop_controller.scroll(clicks)

        elif action == "hover" or action == "mouse_move":
            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                action_description = f"I moved mouse to ({int(tgt_x)}, {int(tgt_y)})."
                await self.desktop_controller.move_mouse(int(tgt_x), int(tgt_y))

        elif action == "drag":
            if "coordinate" in args:
                tgt_x, tgt_y = args["coordinate"]
                action_description = f"I dragged to ({int(tgt_x)}, {int(tgt_y)})."
                await self.desktop_controller.drag(int(tgt_x), int(tgt_y))

        elif action == "sleep" or action == "wait":
            duration = args.get("duration", 3.0)
            duration = args.get("time", duration)
            action_description = f"I waited {duration} seconds."
            await self.desktop_controller.sleep(duration)

        elif action == "stop" or action == "terminate":
            action_description = args.get("thoughts", "Task completed.")
            is_stop_action = True

        else:
            self.logger.warning(f"Unknown action: {action}")
            action_description = f"Unknown action: {action}"

        # Capture new screenshot
        self._num_actions += 1
        if self.save_screenshots:
            await self.desktop_controller.get_screenshot(
                path=os.path.join(self.downloads_folder, f"screenshot{self._num_actions}.png")
            )

        return is_stop_action, action_description

    async def close(self) -> None:
        """Cleanup resources."""
        self.logger.info("Computer Use Agent closed")
