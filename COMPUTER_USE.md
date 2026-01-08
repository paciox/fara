# FARA Computer Use Mode

Desktop automation with vision-guided AI control.

## Overview

Computer Use mode extends FARA beyond browser automation to control your entire desktop using PyAutoGUI. The AI can:

- **Control any application** (Chrome with extensions, Excel, Photoshop, etc.)
- **Navigate Windows** using vision + mouse/keyboard
- **Automate desktop workflows** (file management, app interaction, etc.)
- **Use real Chrome** with your extensions, not Playwright's embedded browser

## Installation

```bash
# Install FARA with PyAutoGUI
pip install -e .

# No browser installation needed for computer use mode!
```

## Quick Start

### Using LM Studio (Recommended)

1. **Start LM Studio** with FARA-7B GGUF model loaded
   - Set context length to **15000+ tokens**
   - Set temperature to **0**
   - Start local server (default: port 1234)

2. **Run Computer Use Mode**

```bash
# Desktop automation
fara-cli --computer_use \
  --task "open Chrome and search for Python tutorials" \
  --base_url http://localhost:1234/v1

# With screenshot saving
fara-cli --computer_use \
  --task "create a new Excel file and add some data" \
  --base_url http://localhost:1234/v1 \
  --save_screenshots \
  --downloads_folder ./screenshots
```

## Example Use Cases

### Browse with Chrome Extensions
```bash
fara-cli --computer_use \
  --task "open Chrome, enable my ad blocker, and browse news sites"
```

### Excel Automation
```bash
fara-cli --computer_use \
  --task "create a spreadsheet with quarterly sales data and format it nicely"
```

### File Management
```bash
fara-cli --computer_use \
  --task "organize my downloads folder by file type"
```

### Multi-App Workflows
```bash
fara-cli --computer_use \
  --task "download a CSV from my browser, open it in Excel, create a chart, and save as PDF"
```

## Safety Features

### Failsafe
**Move your mouse to the top-left corner** to immediately abort execution.

### Pause Between Actions
Default 0.3s pause between actions to prevent runaway automation.

### Visual Feedback
Watch the agent work in real-time as it controls your desktop.

## Supported Actions

The agent can perform these desktop actions:

- `click` - Click at coordinates
- `right_click` - Right-click at coordinates  
- `double_click` - Double-click at coordinates
- `type` - Type text at cursor
- `input_text` - Click location then type
- `keypress` - Press keyboard keys (supports hotkeys like Ctrl+C)
- `scroll` - Scroll up/down
- `hover` / `mouse_move` - Move mouse without clicking
- `drag` - Click and drag
- `wait` / `sleep` - Wait for duration
- `stop` / `terminate` - Complete the task

## Configuration

### CLI Arguments

```bash
fara-cli --computer_use \
  --task "your task here" \
  --base_url http://localhost:1234/v1 \  # LM Studio URL
  --api_key lm-studio \                   # API key (dummy for local)
  --model "fara-7b" \                     # Model name
  --max_rounds 50 \                       # Max action rounds
  --save_screenshots \                    # Save screenshots
  --downloads_folder ./output             # Output folder
```

### Config File

Create `endpoint_configs/lmstudio_config.json`:

```json
{
    "model": "bartowski/microsoft_fara-7b-GGUF",
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio"
}
```

Then run:

```bash
fara-cli --computer_use \
  --task "your task" \
  --endpoint_config endpoint_configs/lmstudio_config.json
```

## Differences from Browser Mode

| Feature | Browser Mode | Computer Use Mode |
|---------|--------------|-------------------|
| Target | Web pages only | Entire desktop |
| Browser | Playwright (embedded) | Any browser (real Chrome, Firefox, etc.) |
| Extensions | ❌ Not supported | ✅ Full support |
| Desktop Apps | ❌ No | ✅ Excel, Photoshop, etc. |
| Precision | High (DOM selectors) | Medium (vision only) |
| Speed | Faster | Slower (real mouse movements) |
| Safety | Sandboxed | ⚠️ Full system access |

## Tips for Best Results

1. **Use high-contrast themes** - easier for vision model to see elements
2. **Maximize windows** - gives AI full view of the application
3. **Close distractions** - minimize popups and notifications
4. **Clear instructions** - be specific about what you want
5. **Monitor first runs** - watch the agent to ensure it's on track

## Troubleshooting

### "Command not found: fara-cli"
```bash
# Use module syntax instead
python -m fara.run_fara --computer_use --task "your task"
```

### Mouse movements too fast
Edit `src/fara/desktop/desktop_controller.py`:
```python
pyautogui.PAUSE = 0.5  # Increase from 0.3
```

### Actions not precise enough
- Increase your screen resolution
- Use larger UI elements
- Ensure good contrast

### Agent clicking wrong things
- The model uses vision only - ensure clear visual separation
- Try being more specific in your task description
- Save screenshots to see what the model sees

## Advanced Usage

### Programmatic API

```python
from fara.computer_use_agent import ComputerUseAgent

config = {
    "model": "fara-7b",
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio"
}

agent = ComputerUseAgent(
    client_config=config,
    max_rounds=50,
    save_screenshots=True,
    downloads_folder="./output"
)

await agent.initialize()
final_answer, actions, observations = await agent.run(
    "open notepad and write a haiku about AI"
)
await agent.close()
```

## Security Warning

⚠️ **Computer Use mode has full access to your desktop.**

- Can open any application
- Can access files
- Can use your browser with your logged-in accounts
- Can execute keyboard shortcuts

**Only use with trusted models and tasks. Monitor execution closely.**

## Comparison with Anthropic Computer Use

FARA Computer Use is similar to [Anthropic's Computer Use](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) but:

- ✅ **Works with open models** (FARA-7B, not just Claude)
- ✅ **Runs locally** (LM Studio, no API costs)
- ✅ **Quantized support** (run on consumer GPUs)
- ❌ Smaller model (7B vs Claude's larger models)
- ❌ May be less accurate for complex tasks

## License

Same as FARA - MIT License
