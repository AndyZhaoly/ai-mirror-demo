"""
IDM-VTON Virtual Try-On Demo
Standalone Gradio app for virtual clothing try-on using IDM-VTON.
"""
import os
import glob
import traceback
import gradio as gr
from PIL import Image

from idm_vton_client import IDMVTONClient

# Service configuration via environment variable
idm_vton_client = IDMVTONClient(os.getenv("VTON_URL", "http://localhost:8001"))

# Directory of segmented clothes extracted by the main demo
EXTRACTED_DIR = "./extracted_clothes"


def get_extracted_clothes():
    """Return list of (display_name, abs_path) for extracted clothes."""
    clothes_list = []
    try:
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for f in sorted(glob.glob(f"{EXTRACTED_DIR}/{ext}")):
                filename = os.path.basename(f)
                if "detection" not in filename and not filename.startswith("."):
                    clothes_list.append((filename, os.path.abspath(f)))
    except Exception as e:
        print(f"[get_extracted_clothes] Error: {e}")
    return clothes_list


def refresh_clothes_list():
    clothes = get_extracted_clothes()
    choices = [("-- 请选择 --", "")] + clothes
    return gr.Dropdown(choices=choices, value="")


def virtual_try_on_handler(person_image, clothes_image_path, clothes_upload, prompt, steps, guidance, seed, preserve_face):
    """
    Run virtual try-on.

    Person image is required. Clothes can come from either the dropdown
    (path string) or a direct upload (PIL Image); uploaded image takes priority.
    """
    if person_image is None:
        return None, "❌ 请上传人物照片"

    # Clothes source: uploaded PIL takes priority over dropdown path
    clothes_source = clothes_upload if clothes_upload is not None else clothes_image_path

    if not clothes_source:
        return None, "❌ 请选择或上传要试穿的衣服"

    if not idm_vton_client.available:
        return None, "❌ IDM-VTON 服务不可用，请确认服务已启动（VTON_URL 环境变量）"

    try:
        if isinstance(clothes_source, str):
            if not os.path.exists(clothes_source):
                return None, f"❌ 找不到服装图片：{clothes_source}"
            clothes_image = Image.open(clothes_source).convert("RGB")
        else:
            clothes_image = clothes_source.convert("RGB")

        person_image = person_image.convert("RGB")

        result_image = idm_vton_client.try_on_images(
            person_image=person_image,
            clothes_image=clothes_image,
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            seed=int(seed),
            preserve_face=preserve_face,
        )

        face_msg = " (已保留原脸)" if preserve_face else ""
        return result_image, f"✅ 虚拟试衣完成！{face_msg}"

    except Exception as e:
        print(traceback.format_exc())
        return None, f"❌ 试衣失败: {str(e)}"


# ========== Gradio UI ==========

with gr.Blocks(title="👗 IDM-VTON 虚拟试衣") as demo:
    gr.Markdown("""
    # 👗 IDM-VTON 虚拟试衣
    ### 上传人物照片 + 选择衣物，AI 生成试穿效果
    """)

    with gr.Row():
        # Left: Inputs
        with gr.Column(scale=1):
            gr.Markdown("#### 📷 人物照片（必填）")
            person_image = gr.Image(
                type="pil",
                label="上传或拍摄人物照片",
                sources=["upload", "webcam"],
            )

            gr.Markdown("#### 👕 选择衣物")
            clothes_dropdown = gr.Dropdown(
                choices=[("-- 请选择 --", "")],
                value="",
                label="从已提取的衣物中选择（来自主演示）",
                interactive=True,
            )
            refresh_btn = gr.Button("🔄 刷新衣物列表", size="sm")

            gr.Markdown("*或上传自定义衣物照片（优先级更高）*")
            clothes_upload = gr.Image(
                type="pil",
                label="上传衣物照片",
                sources=["upload"],
            )

            gr.Markdown("#### ⚙️ 生成参数")
            with gr.Accordion("调整参数", open=False):
                vton_prompt = gr.Textbox(
                    label="提示词",
                    value="a photo of a person wearing clothes",
                    placeholder="描述想要的效果",
                )
                vton_steps = gr.Slider(
                    label="推理步数", minimum=10, maximum=50, value=30, step=1
                )
                vton_guidance = gr.Slider(
                    label="引导系数", minimum=1.0, maximum=5.0, value=2.0, step=0.1
                )
                vton_seed = gr.Number(label="随机种子", value=42, precision=0)
                vton_preserve_face = gr.Checkbox(
                    label="保留原脸 (Face Preservation)",
                    value=True,
                    info="使用 SCHP 模型提取脸部并拼回生成结果",
                )

            vton_btn = gr.Button("✨ 开始试衣", variant="primary", size="lg")

        # Right: Result
        with gr.Column(scale=1):
            gr.Markdown("#### 🎨 试穿结果")
            vton_result = gr.Image(
                type="pil",
                label="生成结果",
                interactive=False,
                height=600,
            )
            vton_status = gr.Textbox(
                label="状态",
                value="等待开始...",
                interactive=False,
            )

    # Event handlers
    refresh_btn.click(fn=refresh_clothes_list, outputs=[clothes_dropdown])

    demo.load(fn=refresh_clothes_list, outputs=[clothes_dropdown])

    vton_btn.click(
        fn=virtual_try_on_handler,
        inputs=[
            person_image,
            clothes_dropdown,
            clothes_upload,
            vton_prompt,
            vton_steps,
            vton_guidance,
            vton_seed,
            vton_preserve_face,
        ],
        outputs=[vton_result, vton_status],
        show_progress=True,
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        theme=gr.themes.Soft(),
    )
