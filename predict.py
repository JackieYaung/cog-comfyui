"""
SDXL Interior Design - ComfyUI based model for Replicate
Uses Depth + Tile ControlNet for room style transformation
"""

import os
import json
import copy
import shutil
import mimetypes
from typing import List
from PIL import Image
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images

os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

mimetypes.add_type("image/webp", ".webp")

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

# 加载室内设计工作流
with open("examples/api_workflows/sdxl_interior_design_api.json", "r", encoding="utf-8") as f:
    WORKFLOW = json.load(f)


class Predictor(BasePredictor):
    def setup(self, weights: str = None):
        """初始化 ComfyUI 服务"""
        for directory in ALL_DIRECTORIES:
            os.makedirs(directory, exist_ok=True)
        os.makedirs(os.environ.get("YOLO_CONFIG_DIR", "/tmp/Ultralytics"), exist_ok=True)

        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def handle_input_file(self, input_file: Path) -> str:
        """处理输入图片，返回文件名"""
        file_extension = os.path.splitext(str(input_file))[1].lower()
        if not file_extension:
            with Image.open(input_file) as img:
                file_extension = f".{img.format.lower()}"
        
        filename = f"input{file_extension}"
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))
        print(f"Input image saved as: {filename}")
        return filename

    def update_workflow(
        self,
        wf: dict,
        image_filename: str,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg: float,
        denoise: float,
        depth_strength: float,
        depth_start: float,
        depth_end: float,
        tile_strength: float,
        tile_start: float,
        tile_end: float,
    ) -> dict:
        """更新工作流参数"""
        # 节点 11: LoadImage - 输入图片
        wf["11"]["inputs"]["image"] = image_filename

        # 节点 5: CLIPTextEncode - 正向提示词
        wf["5"]["inputs"]["text"] = prompt

        # 节点 6: CLIPTextEncode - 负向提示词
        wf["6"]["inputs"]["text"] = negative_prompt

        # 节点 3: KSampler - 采样器参数
        wf["3"]["inputs"]["steps"] = steps
        wf["3"]["inputs"]["cfg"] = cfg
        wf["3"]["inputs"]["denoise"] = denoise

        # 节点 16: ControlNetApplyAdvanced - Depth ControlNet
        wf["16"]["inputs"]["strength"] = depth_strength
        wf["16"]["inputs"]["start_percent"] = depth_start
        wf["16"]["inputs"]["end_percent"] = depth_end

        # 节点 25: ControlNetApplyAdvanced - Tile ControlNet
        wf["25"]["inputs"]["strength"] = tile_strength
        wf["25"]["inputs"]["start_percent"] = tile_start
        wf["25"]["inputs"]["end_percent"] = tile_end

        return wf

    def predict(
        self,
        image: Path = Input(description="输入的室内照片"),
        prompt: str = Input(
            description="设计风格描述",
            default="Professional photography of a mid-century modern living room, architectural digest style. Featuring wood, symmetry, Scandinavian design, mixed textures, vintage, reclaimed furniture, ergonomic layout. Cinematic lighting, atmospheric, photorealistic, sharp focus, 8k, highly detailed.",
        ),
        negative_prompt: str = Input(
            description="负向提示词（不想出现的内容）",
            default="cartoon, illustration, 3d render, painting, drawing, anime, low quality, blurry, watermark, text, signature, people, humans, distorted perspective",
        ),
        steps: int = Input(
            description="推理步数",
            default=20,
            ge=10,
            le=50,
        ),
        cfg: float = Input(
            description="CFG Scale（提示词引导强度）",
            default=6.0,
            ge=1.0,
            le=15.0,
        ),
        denoise: float = Input(
            description="去噪强度（越高变化越大）",
            default=0.75,
            ge=0.0,
            le=1.0,
        ),
        depth_strength: float = Input(
            description="Depth ControlNet 强度（保持空间结构）",
            default=0.35,
            ge=0.0,
            le=1.0,
        ),
        depth_start: float = Input(
            description="Depth ControlNet 开始比例",
            default=0.25,
            ge=0.0,
            le=1.0,
        ),
        depth_end: float = Input(
            description="Depth ControlNet 结束比例",
            default=0.8,
            ge=0.0,
            le=1.0,
        ),
        tile_strength: float = Input(
            description="Tile ControlNet 强度（保持细节纹理）",
            default=0.15,
            ge=0.0,
            le=1.0,
        ),
        tile_start: float = Input(
            description="Tile ControlNet 开始比例",
            default=0.6,
            ge=0.0,
            le=1.0,
        ),
        tile_end: float = Input(
            description="Tile ControlNet 结束比例",
            default=1.0,
            ge=0.0,
            le=1.0,
        ),
        seed: int = Input(
            description="随机种子（留空则随机）",
            default=None,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
    ) -> List[Path]:
        """运行室内设计风格转换"""
        # 清理目录
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # 处理输入图片
        image_filename = self.handle_input_file(image)

        # 复制工作流并更新参数
        wf = copy.deepcopy(WORKFLOW)
        wf = self.update_workflow(
            wf,
            image_filename,
            prompt,
            negative_prompt,
            steps,
            cfg,
            denoise,
            depth_strength,
            depth_start,
            depth_end,
            tile_strength,
            tile_start,
            tile_end,
        )

        # 设置种子
        if seed is not None:
            wf["3"]["inputs"]["seed"] = seed
        else:
            self.comfyUI.randomise_seeds(wf)

        # 连接并运行工作流
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # 获取输出并优化
        output_files = self.comfyUI.get_files([OUTPUT_DIR])
        optimised_files = optimise_images.optimise_image_files(
            output_format, output_quality, output_files
        )

        return [Path(p) for p in optimised_files]
