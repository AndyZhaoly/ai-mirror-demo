# FashionClaw AI Mirror - Implementation Guide

## Project Overview

This is an AI-powered smart wardrobe management system with two main workflows:

1. **Upload & Detect Flow** (NEW): Upload image → Extract clothing → Register to DB → Detect stagnancy → Prompt to sell
2. **Smart Wardrobe Workflow** (EXISTING): Scan DB for stagnant items → Evaluate market price → Match buyer → Execute sale

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASHIONCLAW SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   Gradio UI     │────▶│  GSAM Client    │────▶│  Local GSAM API │       │
│  │   (app.py)      │     │(gsam_client.py) │     │(segment_service)│       │
│  └────────┬────────┘     └─────────────────┘     └─────────────────┘       │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  Database Mgr   │◄───▶│  Workflow Eng.  │◄───▶│   Mock APIs     │       │
│  │(database_mgr.py)│     │  (workflow.py)  │     │ (mock_apis.py)  │       │
│  └────────┬────────┘     └─────────────────┘     └─────────────────┘       │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │  database.json  │                                                        │
│  └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
Demo/
├── app.py                      # Gradio UI - Main entry point
├── workflow.py                 # LangGraph workflow engine
├── database_manager.py         # Database operations (NEW)
├── gsam_client.py             # Grounded-SAM API client
├── segment_service.py         # Local GSAM FastAPI service
├── mock_apis.py               # Mock marketplace & logistics APIs
├── database.json              # JSON database (auto-generated)
├── extracted_clothes/         # Extracted clothing images folder
└── IMPLEMENTATION_GUIDE.md    # This file
```

## Key Files Explained

### 1. database_manager.py (NEW)

**Purpose**: Handles all database operations with stagnancy tracking.

**Key Functions**:
- `add_item(name, clothing_type, image_path, ...)` - Register new clothing item
  - Auto-generates mock price: ¥100-500 for upper, ¥80-400 for lower
  - Auto-generates mock purchase_date: ~400 days ago (triggers >365 detection)
- `is_stagnant(item, threshold_days=365)` - Check if item is stagnant
- `get_stagnant_items()` - Get all items not worn for 365+ days
- `update_item_status(item_id, new_status)` - Update item status

**Schema**:
```json
{
  "item_id": "ABC12345",
  "name": "Blue T-Shirt",
  "clothing_type": "upper",
  "date_added": "2024-04-04T10:30:00",
  "purchase_date": "2023-02-15",
  "status": "in_closet",
  "original_price": 299,
  "image": "extracted_clothes/upper_0_0404_103000.png",
  "extracted_from": "temp_upload.jpg"
}
```

### 2. workflow.py (MODIFIED)

**Purpose**: LangGraph workflow engine with two entry points.

**Original Flow**:
```
monitor_node → evaluate_node → wait_for_user_node → execute_node
     ↑                                              ↓
   Scan DB                                      Update DB
```

**New Upload Flow**:
```
stagnancy_check_node → evaluate_node → wait_for_user_node → execute_node
        ↑                                                    ↓
   Check >365 days                                      Update DB
```

**Key Functions**:
- `run_workflow_until_user_input()` - Original scanner flow
- `run_upload_workflow_until_user_input(item)` - New upload flow for single item
- `stagnancy_check_node(state)` - NEW: Checks if uploaded item is stagnant
- `resume_workflow(user_approved)` - Continue after user decision

### 3. gsam_client.py (UNCHANGED)

**Purpose**: Client for Grounded-SAM segmentation API.

**Key Functions**:
- `extract_upper_body(image_path)` - Extract upper body clothing
- `extract_lower_body(image_path)` - Extract lower body clothing
- `extract_both(image_path)` - Extract both upper and lower

**API Endpoints Called**:
- `POST /extract_upper_body` - Returns base64 encoded segmented images
- `POST /extract_lower_body` - Returns base64 encoded segmented images

**Configuration**:
```python
GSAM_SERVICE_URL = "http://localhost:8000"  # Set via env var
```

### 4. segment_service.py (UNCHANGED - Run this first!)

**Purpose**: FastAPI service that runs SAM model locally.

**How to Run**:
```bash
# In the Demo directory
conda activate gsam_env  # or your environment with SAM installed
python segment_service.py
```

**Endpoints**:
- `GET /health` - Health check
- `POST /extract_upper_body` - Returns upper body masks
- `POST /extract_lower_body` - Returns lower body masks

**Requirements**: SAM model checkpoint at `/home/zhaoliyang/Grounded-Segment-Anything/sam_vit_h_4b8939.pth`

### 5. mock_apis.py (UNCHANGED)

**Purpose**: Simulates external marketplace and logistics APIs.

**Functions**:
- `check_market_price(item_name)` - Returns mock market price (¥50-450)
- `get_buyer_offer(market_price)` - Returns mock buyer with offer
- `check_buyer_credit(buyer_id)` - Returns mock credit rating
- `execute_logistics(item_name, buyer_name)` - Returns tracking info
- `update_item_status(item_id, new_status)` - Updates database.json

### 6. app.py (MODIFIED)

**Purpose**: Gradio UI with two tabs.

**Tab 1: "📤 上传&检测" (NEW)**
```
User uploads image
    ↓
save as temp_upload.jpg
    ↓
gsam_client.extract_upper_body() → save to extracted_clothes/upper_*.png
    ↓
gsam_client.extract_lower_body() → save to extracted_clothes/lower_*.png
    ↓
database_manager.add_item() for each extracted piece (auto-mock dates/prices)
    ↓
Check is_stagnant() for each item
    ↓
If stagnant: run_upload_workflow_until_user_input()
    ↓
Display: Market price, buyer offer, credit rating
    ↓
User clicks ✅ Confirm or ❌ Reject
    ↓
resume_workflow() → execute sale or keep item
```

**Tab 2: "🤖 智能衣橱工作流" (EXISTING)**
```
Click "启动智能衣橱系统"
    ↓
run_workflow_until_user_input()
    ↓
Scan database.json for items with last_worn_days_ago > 365
    ↓
Evaluate → Show prompt → User decides → Execute
```

## How to Run

### Step 1: Start the GSAM Service (Required!)

```bash
cd "/Users/liyangzhao/Desktop/智能镜/demo"
conda activate gsam_env  # Environment with SAM installed
python segment_service.py
# Should show: "Uvicorn running on http://0.0.0.0:8000"
```

### Step 2: Start the Main Application

In a new terminal:

```bash
cd "/Users/liyangzhao/Desktop/智能镜/demo"
conda activate fashionclaw  # or your main environment
python app.py
# Should show: "Running on local URL: http://127.0.0.1:7860"
```

### Step 3: Use the Interface

**New Upload Flow**:
1. Open browser to `http://127.0.0.1:7860`
2. Click "📤 上传&检测" tab
3. Upload a photo of yourself wearing clothes
4. (Optional) Enter a name prefix like "我的"
5. Click "🚀 上传并检测"
6. System will:
   - Extract upper and lower body clothing
   - Register them to database.json with mock dates (~400 days ago)
   - Check if >365 days old (will always trigger for demo)
   - Show sell prompt with market evaluation
7. Click "✅ 确认出售" or "❌ 拒绝出售"
8. View transaction result

**Original Scanner Flow**:
1. Click "🤖 智能衣橱工作流" tab
2. Click "🚀 启动智能衣橱系统"
3. System scans existing database for stagnant items
4. Follow prompts to sell or keep

## Database Schema Details

### New Schema (for uploaded items)

```json
{
  "wardrobe": [
    {
      "item_id": "ABC12345",
      "name": "我的_上衣_1",
      "clothing_type": "upper",
      "date_added": "2024-04-04T10:30:00",
      "purchase_date": "2023-02-15",
      "status": "in_closet",
      "original_price": 299,
      "image": "extracted_clothes/upper_0_0404_103000.png",
      "extracted_from": "temp_upload.jpg"
    }
  ]
}
```

### Old Schema (for demo items)

```json
{
  "wardrobe": [
    {
      "item_id": "001",
      "name": "Blue Denim Jacket",
      "last_worn_days_ago": 420,
      "status": "in_closet",
      "original_price": 299,
      "image": "images/denim_jacket.jpg"
    }
  ]
}
```

**Note**: Both schemas work simultaneously. Old items use `last_worn_days_ago`, new items use `purchase_date`.

## Mock Data Strategy

For demonstration purposes, the system automatically generates:

1. **Purchase Date**: ~400 days ago (random between 380-450 days)
   - Ensures items always trigger the >365 day stagnancy check
   - Code: `generate_mock_purchase_date()` in database_manager.py

2. **Original Price**: Based on clothing type
   - Upper body: ¥100-500
   - Lower body: ¥80-400
   - Code: `generate_mock_price()` in database_manager.py

3. **Market Evaluation**: Determined by mock_apis.py
   - Market price: 50-70% of original
   - Buyer offer: 80-95% of market price
   - Credit rating: Weighted random (50% Excellent, 35% Good, etc.)

## Troubleshooting

### GSAM Service Not Running

**Error**: `ConnectionError: Cannot connect to Grounded-SAM service at http://localhost:8000`

**Solution**:
```bash
# Check if service is running
curl http://localhost:8000/health

# If not, start it
python segment_service.py
```

### SAM Model Not Found

**Error**: `FileNotFoundError: sam_vit_h_4b8939.pth`

**Solution**: Update path in `segment_service.py`:
```python
SAM_CHECKPOINT_PATH = "/path/to/your/sam_vit_h_4b8939.pth"
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**: Kill existing process or change port:
```bash
# Find and kill
lsof -ti:8000 | xargs kill -9
# Or change port in segment_service.py
uvicorn.run(app, host="0.0.0.0", port=8001)
```

## Code Modification Summary

| File | Lines Added | Purpose |
|------|-------------|---------|
| database_manager.py | ~200 | NEW: Database operations with stagnancy tracking |
| workflow.py | ~50 | MODIFIED: Added stagnancy_check_node and upload flow |
| app.py | ~150 | MODIFIED: Added "📤 上传&检测" tab and handlers |

## Next Steps for Extension

To connect to a real Grounded-SAM API instead of local service:

1. Modify `gsam_client.py`:
```python
# Change this
GSAM_SERVICE_URL = os.getenv("GSAM_SERVICE_URL", "http://localhost:8000")
# To this
GSAM_SERVICE_URL = "https://your-external-api.com"
```

2. Add authentication headers if needed:
```python
headers = {"Authorization": "Bearer YOUR_API_KEY"}
response = requests.post(url, files=files, data=data, headers=headers)
```

3. The rest of the code remains unchanged - it already uses HTTP API calls.

## Contact

For questions about this implementation, refer to:
- `database_manager.py` for DB operations
- `workflow.py` for flow logic
- `app.py` for UI handlers
