"""
服装视觉搜索 - 支持 SerpAPI Google Reverse Image 和 Bing 搜索
默认使用 SerpAPI + 图床（更稳定），失败时回退到 Bing
"""
import os
import time
import re
import requests
from typing import Optional, Dict, List


def upload_to_catbox(image_path: str) -> Optional[str]:
    """
    上传图片到 catbox.moe 临时图床

    Args:
        image_path: 本地图片路径

    Returns:
        公开访问的 URL 或 None
    """
    try:
        with open(image_path, 'rb') as f:
            data = {'reqtype': 'fileupload', 'time': '1h'}  # 1小时过期
            files = {'fileToUpload': f}
            response = requests.post(
                'https://litterbox.catbox.moe/resources/internals/api.php',
                data=data,
                files=files,
                timeout=30
            )

        if response.status_code == 200 and response.text.startswith('http'):
            return response.text.strip()
        return None
    except Exception as e:
        print(f"[Catbox] 上传失败: {e}")
        return None


def search_clothing_with_serpapi(image_path: str) -> Dict:
    """
    使用 SerpAPI 的 Google Reverse Image 进行以图搜图
    需要 SERPAPI_KEY 环境变量

    Args:
        image_path: 本地图片路径

    Returns:
        {
            "success": bool,
            "brand": str or None,
            "title": str or None,
            "price": str or None,
            "source": str or None,
            "matches": list,
            "raw_text": str
        }
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return {"success": False, "error": "SERPAPI_KEY not set", "raw_text": ""}

    if not os.path.exists(image_path):
        return {"success": False, "error": "图片不存在", "raw_text": ""}

    try:
        from serpapi import GoogleSearch
    except ImportError:
        return {"success": False, "error": "serpapi not installed", "raw_text": ""}

    # 步骤1: 上传图片到图床
    print(f"[SerpAPI] 上传图片到图床: {image_path}")
    image_url = upload_to_catbox(image_path)
    if not image_url:
        return {"success": False, "error": "图床上传失败", "raw_text": ""}

    print(f"[SerpAPI] 图床URL: {image_url}")

    # 保存调试图床URL到文件，方便用户对比
    try:
        debug_file = "/tmp/last_search_image_url.txt"
        with open(debug_file, "w") as f:
            f.write(f"图床URL: {image_url}\n")
            f.write(f"原图路径: {image_path}\n")
            f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"[SerpAPI] 调试信息已保存: {debug_file}")
    except Exception as e:
        print(f"[SerpAPI] 保存调试信息失败: {e}")

    # 步骤2: 调用 SerpAPI Google Reverse Image
    try:
        print("[SerpAPI] 调用 Google Reverse Image...")
        params = {
            "engine": "google_reverse_image",
            "image_url": image_url,
            "api_key": api_key,
            "hl": "zh-CN",
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        # 解析结果
        output = {
            "success": True,
            "brand": None,
            "title": None,
            "price": None,
            "source": "Google Reverse Image (SerpAPI)",
            "matches": [],
            "raw_text": ""
        }

        if "error" in results:
            return {"success": False, "error": results["error"], "raw_text": ""}

        # 提取图片结果
        if "image_results" in results:
            matches = results["image_results"][:10]  # 取前10个
            output["matches"] = matches

            # 收集所有文本用于分析
            all_titles = []

            for match in matches:
                title = match.get("title", "")
                source = match.get("source", "")

                if title:
                    all_titles.append(title)

                    # 提取品牌
                    if not output["brand"]:
                        brands = ['Nike', 'Adidas', 'Zara', 'H&M', 'Uniqlo', 'Gucci', 'Prada', 'Chanel',
                                 'Louis Vuitton', 'Supreme', 'Balenciaga', 'Dior', 'Burberry', 'Versace',
                                 'Ted Baker', 'Calvin Klein', 'Tommy Hilfiger', 'Ralph Lauren',
                                 '优衣库', '耐克', '阿迪达斯', 'ZARA', '古驰', '普拉达', '香奈儿']
                        for brand in brands:
                            if brand.lower() in title.lower():
                                output["brand"] = brand
                                break

                    # 提取价格
                    if not output["price"]:
                        price_patterns = [
                            r'¥\s*([\d,]+)',
                            r'\$\s*([\d,]+)',
                            r'([\d,]+)\s*元',
                            r'USD\s*([\d,]+)',
                            r'£\s*([\d,]+)',
                            r'€\s*([\d,]+)',
                        ]
                        for pattern in price_patterns:
                            match_price = re.search(pattern, title)
                            if match_price:
                                output["price"] = match_price.group(0)
                                break

                    # 设置标题（第一个有意义的标题）
                    if not output["title"] and len(title) > 10:
                        output["title"] = title[:100]

            output["raw_text"] = "\n".join(all_titles[:5])

        print(f"[SerpAPI] 结果: brand={output['brand']}, price={output['price']}, matches={len(output['matches'])}")

        # 保存调试信息到文件
        try:
            debug_file = "/tmp/last_google_lens_result.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                import json
                json.dump({
                    "image_url": image_url,
                    "search_result": results,
                    "parsed_output": output,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
            print(f"[SerpAPI] 完整结果已保存: {debug_file}")
        except Exception as e:
            print(f"[SerpAPI] 保存调试信息失败: {e}")

        return output

    except Exception as e:
        print(f"[SerpAPI] 搜索失败: {e}")
        return {"success": False, "error": str(e), "raw_text": ""}


def search_clothing_on_bing(image_path: str, headless: bool = True, timeout: int = 30) -> Dict:
    """
    使用 Bing 视觉搜索查找相似服装（备用方案）

    Returns:
        {
            "success": bool,
            "brand": str or None,
            "title": str or None,
            "price": str or None,
            "source": str or None,
            "raw_text": str
        }
    """
    from playwright.sync_api import sync_playwright, Browser

    if not os.path.exists(image_path):
        return {"success": False, "error": "图片不存在", "raw_text": ""}

    browser: Optional[Browser] = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )

            context = browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            )

            page = context.new_page()

            # 访问 Bing 视觉搜索
            print("[Bing] 打开视觉搜索...")
            page.goto("https://www.bing.com/visualsearch", wait_until="networkidle", timeout=timeout*1000)
            time.sleep(2)

            # 上传图片
            print(f"[Bing] 上传图片...")
            file_input = page.locator('input[type="file"]').first
            file_input.set_input_files(image_path)

            # 等待结果
            print("[Bing] 等待分析...")
            time.sleep(8)

            # 抓取结果
            results = {
                "success": True,
                "brand": None,
                "title": None,
                "price": None,
                "source": "Bing Visual Search",
                "matches": [],
                "raw_text": ""
            }

            # 提取页面文本
            try:
                page_text = page.inner_text('body', timeout=5000)

                # 常见品牌词匹配
                brands = ['Nike', 'Adidas', 'Zara', 'H&M', 'Uniqlo', 'Gucci', 'Prada', 'Chanel',
                         'Louis Vuitton', 'Supreme', 'Balenciaga', 'Dior', 'Burberry',
                         '优衣库', '耐克', '阿迪达斯', 'ZARA']
                for brand in brands:
                    if brand.lower() in page_text.lower():
                        results["brand"] = brand
                        break

                # 价格匹配
                price_patterns = [
                    r'¥\s*\d+',
                    r'\$\s*\d+',
                    r'\d+\s*元',
                    r'USD\s*\d+',
                ]
                for pattern in price_patterns:
                    match = re.search(pattern, page_text)
                    if match:
                        results["price"] = match.group(0)
                        break

                # 提取所有文本行
                lines = [l.strip() for l in page_text.split('\n')
                        if l.strip() and 10 < len(l.strip()) < 150]
                results["raw_text"] = '\n'.join(lines[:50])

                # 尝试找产品标题
                for line in lines:
                    if any(keyword in line.lower() for keyword in
                           ['shirt', 'jacket', 'dress', 'hoodie', 'sweater', 't-shirt', '外套', '衬衫', '卫衣']):
                        if not results["title"] or len(line) > len(results["title"]):
                            results["title"] = line[:100]

            except Exception as e:
                print(f"[Bing] 提取文本失败: {e}")

            browser.close()
            return results

    except Exception as e:
        if browser:
            try:
                browser.close()
            except:
                pass
        return {"success": False, "error": str(e), "raw_text": ""}


def search_clothing_on_google(image_path: str) -> Dict:
    """
    主搜索函数：优先使用 SerpAPI，失败时回退到 Bing

    Args:
        image_path: 图片本地路径

    Returns:
        搜索结果字典
    """
    # 优先尝试 SerpAPI
    if os.getenv("SERPAPI_KEY"):
        result = search_clothing_with_serpapi(image_path)
        if result.get("success"):
            return result
        print("[Search] SerpAPI 失败，回退到 Bing...")

    # 回退到 Bing
    return search_clothing_on_bing(image_path)


# 保持向后兼容的别名
search_clothing = search_clothing_on_google
search_clothing_on_bing = search_clothing_on_bing


if __name__ == "__main__":
    # 测试
    from PIL import Image

    test_image = "/tmp/test_search.jpg"
    if not os.path.exists(test_image):
        img = Image.new('RGB', (400, 500), color='red')
        img.save(test_image)

    print("测试服装视觉搜索...")
    result = search_clothing_on_google(test_image)
    print(f"结果: {result}")
