# Computer Use System

Multi-agent orchestration system for desktop automation tasks using FARA.

## Features

- **Multi-Agent Orchestration**: Coordinates planner, executor, and critic agents
- **Action Loop Detection**: Prevents infinite repetition of failed actions  
- **Screenshot Debugging**: Saves screenshots after each action
- **Flexible Configuration**: Support for local LM Studio or remote endpoints
- **Trajectory Tracking**: Records all actions and observations

## Quick Start

### Using the Orchestrated System

```bash
# Run with the Computer Use System (via webeval)
python webeval/scripts/run_computer_use.py \
  --task "click the Chrome icon in the taskbar" \
  --output_dir ./outputs \
  --max_rounds 10
```

### Using Direct CLI (Single Agent)

```bash
# Run with direct CLI (simpler, single agent)
fara-cli --computer_use \
  --task "open Chrome and search for Python tutorials" \
  --save_screenshots \
  --downloads_folder ./screenshots \
  --max_rounds 20
```

## Architecture

### Single Agent Mode (Default)
```
User Task → Executor Agent (FARA-7B) → Actions → Desktop
```

### Multi-Agent Mode (Planned)
```
User Task → Planner (GPT-4) → Step Plan
              ↓
         Executor (FARA-7B) → Execute Step → Desktop
              ↓
          Critic (GPT-4) → Evaluate Success
              ↓
         Planner → Re-plan if needed
```

## Configuration

### Executor Config (FARA-7B)
```json
{
  "model": "microsoft/Fara-7B",
  "base_url": "http://localhost:1234/v1",
  "api_key": "lm-studio"
}
```

### Example Usage with Config File

```bash
# Create config
cat > executor_config.json << EOF
{
  "model": "microsoft/Fara-7B", 
  "base_url": "http://localhost:1234/v1",
  "api_key": "lm-studio"
}
EOF

# Run task
python webeval/scripts/run_computer_use.py \
  --task "open notepad and type hello world" \
  --config executor_config.json \
  --output_dir ./outputs
```

## API Usage

```python
from webeval.systems import ComputerUseSystem

# Create system
system = ComputerUseSystem(
    system_name="my_computer_use",
    executor_client_cfg={
        "model": "microsoft/Fara-7B",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
    },
    max_rounds=20,
    save_screenshots=True,
    downloads_folder="./outputs",
)

# Run task
trajectory = system.get_answer(
    question_id="task1",
    example_data={"task": "open Chrome"},
    output_dir="./outputs",
)

print(f"Final answer: {trajectory.final_answer.answer}")
print(f"Actions taken: {len(trajectory.action)}")
```

## Output Structure

```
outputs/
└── task_cli_task/
    ├── screenshot0.png
    ├── screenshot1.png
    ├── screenshot2.png
    └── trajectory.json
```

## Tips for Better Results

### 1. Use Simple, Concrete Tasks
❌ Bad: "browse the web and find Python tutorials"  
✅ Good: "click Chrome icon, type 'python tutorials', press enter"

### 2. Break Complex Tasks Down
Instead of one big task:
```bash
# Step 1
--task "open Chrome browser"

# Step 2  
--task "type 'python tutorials' in search box and press enter"
```

### 3. Prefer Keyboard Over Mouse
The quantized model struggles with coordinate accuracy:
❌ "click the Chrome icon at position X,Y"  
✅ "press windows key, type 'chrome', press enter"

### 4. Use Loop Detection
The system automatically stops after 3 repeated identical actions.

## Troubleshooting

### Model Keeps Clicking Wrong Location
- **Cause**: Quantized model has reduced spatial reasoning
- **Fix**: Use keyboard-based actions or simpler coordinate targets

### Action Loops
- **Cause**: Model doesn't recognize task completion
- **Fix**: Loop detection will auto-stop after 3 repeats

### Multi-Monitor Issues
- **Cause**: Screenshots only capture primary monitor
- **Fix**: Move target application to primary monitor

## Future Enhancements

- [ ] Full multi-agent orchestration with GPT-4 planner/critic
- [ ] Adaptive action planning based on feedback
- [ ] Better vision-language grounding for coordinates
- [ ] Support for secondary monitor selection
- [ ] Integration with AutoGen MagenticOne agents
