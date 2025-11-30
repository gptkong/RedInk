"""Jm2Api 图片生成器（URL 响应格式）"""
import logging
import time
import random
import requests
from typing import Dict, Any, Optional, List
from .base import ImageGeneratorBase

logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, base_delay: float = 2):
    """错误重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"请求失败，{delay:.1f}秒后重试 (尝试 {attempt + 2}/{max_retries}): {str(e)[:100]}")
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


class Jm2ApiGenerator(ImageGeneratorBase):
    """
    Jm2Api 图片生成器
    
    适用于返回 URL 格式的图片生成 API：
    {"created": 1759058768, "data": [{"url": "https://example.com/image.jpg"}]}
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.debug("初始化 Jm2ApiGenerator...")
        self.base_url = config.get('base_url', 'https://api.example.com').rstrip('/').rstrip('/v1')
        self.model = config.get('model', 'default-model')
        self.default_ratio = config.get('default_ratio', '3:4')
        self.default_resolution = config.get('default_resolution', '2k')

        logger.info(f"Jm2ApiGenerator 初始化完成: base_url={self.base_url}, model={self.model}")

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        if not self.api_key:
            logger.error("Jm2Api API Key 未配置")
            raise ValueError(
                "Jm2Api API Key 未配置。\n"
                "解决方案：在系统设置页面编辑该服务商，填写 API Key"
            )
        return True

    def get_supported_sizes(self) -> List[str]:
        """获取支持的分辨率"""
        return ["1k", "2k", "4k"]

    def get_supported_aspect_ratios(self) -> List[str]:
        """获取支持的宽高比"""
        return ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"]

    @retry_on_error(max_retries=3, base_delay=2)
    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = None,
        resolution: str = None,
        model: str = None,
        **kwargs
    ) -> bytes:
        """
        生成图片

        Args:
            prompt: 图片描述
            aspect_ratio: 宽高比 (ratio)
            resolution: 分辨率
            model: 模型名称

        Returns:
            生成的图片二进制数据
        """
        self.validate_config()

        if model is None:
            model = self.model
        if aspect_ratio is None:
            aspect_ratio = self.default_ratio
        if resolution is None:
            resolution = self.default_resolution

        logger.info(f"Jm2Api 生成图片: model={model}, ratio={aspect_ratio}, resolution={resolution}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "prompt": prompt
        }

        # ratio 和 resolution 必须同时出现，或者都不出现
        # 只有当不是默认值 (1:1, 2k) 时才添加这两个参数
        if aspect_ratio != "1:1" or resolution != "2k":
            payload["ratio"] = aspect_ratio
            payload["resolution"] = resolution

        api_url = f"{self.base_url}/v1/images/generations"
        logger.debug(f"  请求体: {payload}")
        logger.debug(f"  发送请求到: {api_url}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)

        if response.status_code != 200:
            error_detail = response.text[:500]
            status_code = response.status_code
            logger.error(f"Jm2Api 请求失败: status={status_code}, error={error_detail}")

            if status_code == 401:
                raise Exception(
                    "❌ API Key 认证失败\n\n"
                    "【可能原因】\n"
                    "1. API Key 无效或已过期\n"
                    "2. API Key 格式错误\n\n"
                    "【解决方案】\n"
                    "在系统设置页面检查 API Key 是否正确"
                )
            elif status_code == 429:
                raise Exception(
                    "⏳ API 配额或速率限制\n\n"
                    "【解决方案】\n"
                    "1. 稍后再试\n"
                    "2. 检查 API 配额使用情况"
                )
            else:
                raise Exception(
                    f"❌ Jm2Api 请求失败 (状态码: {status_code})\n\n"
                    f"【错误详情】\n{error_detail[:300]}\n\n"
                    f"【请求地址】{api_url}\n"
                    f"【模型】{model}"
                )

        result = response.json()
        data = result.get('data') or []
        logger.debug(f"  API 响应: data 长度={len(data)}")

        if len(data) > 0:
            item = data[0]
            if "url" in item:
                image_url = item["url"]
                logger.info(f"从 URL 下载图片: {image_url[:100]}...")
                return self._download_image(image_url)

        logger.error(f"无法从响应中提取图片数据: {str(result)[:200]}")
        raise Exception(
            f"图片数据提取失败：未找到 url 数据。\n"
            f"API响应片段: {str(result)[:500]}\n"
            "可能原因：\n"
            "1. API返回格式与预期不符\n"
            "2. 该模型不支持图片生成\n"
            "建议：检查API文档确认返回格式要求"
        )

    def _download_image(self, url: str) -> bytes:
        """下载图片并返回二进制数据"""
        logger.info(f"下载图片: {url[:100]}...")
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                logger.info(f"✅ 图片下载成功: {len(response.content)} bytes")
                return response.content
            else:
                raise Exception(f"下载图片失败: HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            raise Exception("❌ 下载图片超时，请重试")
        except Exception as e:
            raise Exception(f"❌ 下载图片失败: {str(e)}")
