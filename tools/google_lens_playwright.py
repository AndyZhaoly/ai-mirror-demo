"""
Google Lens 搜索工具 - 使用 Playwright 抓取相似商品信息
FashionClaw 价格查询模块
"""
import os
import time
import re
from typing import Optional, Dict, List
from playwright.sync_api import sync_playwright, Page, Browser
# playwright_stealth 暂时不用，避免兼容性问题
# from playwright_stealth import stealth


def search_clothing_on_google_lens(image_path: str, headless: bool = False, timeout: int = 30) -> str:
    """
    使用 Google Lens 搜索相似服装并抓取结果

    Args:
        image_path: 本地图片路径
        headless: 是否无头模式（调试时用 False）
        timeout: 等待超时时间（秒）

    Returns:
        抓取到的文本信息
    """
    if not os.path.exists(image_path):
        return f"错误：图片不存在 {image_path}"

    result_texts = []
    browser: Optional[Browser] = None

    try:
        with sync_playwright() as p:
            # 启动浏览器（增强反检测）
            browser = p.chromium.launch(
                headless=headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-setuid-sandbox',
                    '--disable-accelerated-2d-canvas',
                    '--disable-accelerated-jpeg-decoding',
                    '--disable-accelerated-mjpeg-decode',
                    '--disable-accelerated-video-decode',
                    '--disable-app-list-dismiss-on-blur',
                    '--disable-canvas-aa',
                    '--disable-component-update',
                    '--disable-default-apps',
                    '--disable-dev-shm-usage',
                    '--disable-extensions',
                    '--disable-features=TranslateUI',
                    '--disable-hang-monitor',
                    '--disable-ipc-flooding-protection',
                    '--disable-popup-blocking',
                    '--disable-prompt-on-repost',
                    '--disable-renderer-backgrounding',
                    '--disable-sync',
                    '--force-color-profile=srgb',
                    '--metrics-recording-only',
                    '--no-first-run',
                    '--safebrowsing-disable-auto-update',
                    '--enable-automation',  # 注意：这个要加，反而能绕过一些检测
                    '--password-store=basic',
                    '--use-mock-keychain',
                    '--lang=zh-CN',
                ]
            )

            # 创建上下文（更多参数伪装）
            context = browser.new_context(
                viewport={'width': 1440, 'height': 900},  # 常见 MacBook 分辨率
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                locale='zh-CN',
                timezone_id='Asia/Shanghai',
                geolocation={'latitude': 39.9042, 'longitude': 116.4074},  # 北京坐标
                permissions=['geolocation'],
                color_scheme='light',
            )

            # 注入 JavaScript 绕过检测
            context.add_init_script("""
                // 覆盖 webdriver 检测
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });

                // 覆盖 plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });

                // 覆盖 languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['zh-CN', 'zh', 'en-US', 'en']
                });

                // 覆盖 chrome
                window.chrome = {
                    runtime: {},
                    loadTimes: function() {},
                    csi: function() {},
                    app: {}
                };

                // 覆盖 permission
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)

            page = context.new_page()

            # 设置额外 HTTP 头
            page.set_extra_http_headers({
                'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
            })

            # 访问 Google Lens
            print("[Google Lens] 正在打开 Google Lens...")
            page.goto("https://lens.google.com/search?p=", wait_until="domcontentloaded", timeout=timeout*1000)
            time.sleep(2)

            # 检测是否需要验证
            verification_keywords = ["验证", "我不是机器人", "reCAPTCHA", "人机验证", "Identity check", "Sign in", "登录", "账号"]
            page_text = page.inner_text('body')[:500]

            needs_verification = any(keyword in page_text for keyword in verification_keywords)

            if needs_verification:
                print("\n" + "="*60)
                print("⚠️  检测到 Google 安全验证")
                print("="*60)
                print("请在浏览器中完成以下操作：")
                print("1. 如果有验证码，请完成验证")
                print("2. 如果需要登录，请登录 Google 账号（可选）")
                print("3. 完成后，回到此终端按 Enter 键继续...")
                print("="*60)
                input("按 Enter 继续...")

            # 上传图片 - 通过文件选择器
            print(f"[Google Lens] 正在上传图片: {os.path.basename(image_path)}")

            # 等待文件上传按钮出现
            # 策略1: 直接找 input[type=file]
            try:
                file_input = page.locator('input[type="file"]').first
                file_input.set_input_files(image_path)
                print("[Google Lens] 图片已上传，等待分析...")
            except Exception as e:
                print(f"[Google Lens] 上传失败，尝试备用策略: {e}")
                # 策略2: 点击相机图标再上传
                try:
                    camera_btn = page.locator('[aria-label*="camera"], [title*="camera"], button svg').first
                    camera_btn.click()
                    time.sleep(1)
                    file_input = page.locator('input[type="file"]').first
                    file_input.set_input_files(image_path)
                except Exception as e2:
                    return f"上传图片失败: {e2}"

            # 等待搜索结果加载
            time.sleep(3)

            # 再次检测验证（上传后也可能触发）
            page_text = page.inner_text('body')[:500]
            needs_verification = any(keyword in page_text for keyword in verification_keywords)

            if needs_verification:
                print("\n" + "="*60)
                print("⚠️  检测到验证页面")
                print("="*60)
                print("请在浏览器中完成验证，然后按 Enter 继续...")
                input("按 Enter 继续...")

            # 等待结果容器出现（多种策略）
            print("[Google Lens] 等待搜索结果...")

            # 策略1: 等待购物结果/相似商品区域
            selectors = [
                '[data-async-context^="query:"]',  # 搜索结果容器
                '[jsname]',  # Google 的动态内容
                'a[href*="shopping"]',  # 购物链接
                'a[href*="product"]',  # 产品链接
                '[role="listitem"]',  # 列表项
                'img[src*="http"]',  # 结果图片
            ]

            found = False
            for selector in selectors:
                try:
                    page.wait_for_selector(selector, timeout=5000)
                    found = True
                    print(f"[Google Lens] 找到结果元素: {selector}")
                    break
                except:
                    continue

            if not found:
                print("[Google Lens] 警告：未找到标准结果元素，尝试抓取页面文本...")

            # 等待 JS 渲染完成
            time.sleep(2)

            # 再次检测是否需要人工干预
            page_text = page.inner_text('body')[:800]
            has_results = any(keyword in page_text for keyword in ['Visual matches', '匹配结果', '相似商品', 'Shopping', '购买'])
            has_error = any(keyword in page_text for keyword in ['无法访问', '无法加载', '出错', 'expired', 'error'])

            if not has_results or has_error:
                print("\n" + "=" * 60)
                print("⚠️  未检测到搜索结果或页面异常")
                print("=" * 60)
                print("可能的情况：")
                print("1. Google 需要验证（请检查浏览器）")
                print("2. 图片识别失败")
                print("3. 网络问题")
                print("\n请在浏览器中：")
                print("- 完成任何验证")
                print("- 或者重新上传图片")
                print("- 然后按 Enter 继续抓取结果...")
                print("=" * 60)
                input("按 Enter 继续...")

            # 最终确认：给用户时间看结果，然后选择是否抓取
            print("\n" + "=" * 60)
            print("✅ 搜索完成！")
            print("=" * 60)
            print("请在浏览器中查看搜索结果")
            print("如果结果满意，按 Enter 抓取页面内容...")
            print("如果需要重新搜索，请关闭浏览器重新开始")
            print("=" * 60)
            input("按 Enter 开始抓取结果...")

            # 抓取页面文本内容（多策略）
            print("[Google Lens] 正在抓取结果文本...")

            # 策略1: 提取所有链接文本（购物结果通常包含价格）
            try:
                links = page.locator('a').all()
                for link in links[:20]:  # 限制数量
                    text = link.inner_text(timeout=1000).strip()
                    href = link.get_attribute('href') or ''
                    # 过滤有用的文本（包含价格、品牌信息）
                    if text and len(text) > 3 and len(text) < 200:
                        if any(keyword in text.lower() for keyword in ['¥', '$', '元', 'price', 'shop', 'buy']):
                            result_texts.append(f"[链接] {text}")
                        elif any(keyword in href for keyword in ['shopping', 'product', 'item']):
                            result_texts.append(f"[商品] {text}")
            except Exception as e:
                print(f"[Google Lens] 提取链接失败: {e}")

            # 策略2: 提取所有可见文本
            try:
                # 获取页面主要文本内容
                texts = page.locator('body').inner_text()
                # 清理并分割
                lines = [line.strip() for line in texts.split('\n') if line.strip()]
                # 过滤有用的行（长度适中，包含关键词）
                for line in lines[:50]:
                    if len(line) > 5 and len(line) < 150:
                        # 过滤掉 Google 的导航文本
                        if not any(skip in line for skip in ['Google', 'Sign in', 'Search', 'Privacy', 'Terms']):
                            result_texts.append(line)
            except Exception as e:
                print(f"[Google Lens] 提取文本失败: {e}")

            # 策略3: 尝试找特定结构的结果卡片
            try:
                # 寻找可能包含商品信息的 div
                cards = page.locator('div, span, article').all()
                for card in cards[:30]:
                    try:
                        text = card.inner_text(timeout=500).strip()
                        # 如果包含价格符号，很可能是商品
                        if '¥' in text or '$' in text or '元' in text:
                            if len(text) < 100 and text not in result_texts:
                                result_texts.append(f"[价格] {text}")
                    except:
                        pass
            except Exception as e:
                print(f"[Google Lens] 提取卡片失败: {e}")

            # 关闭浏览器
            browser.close()

            # 去重并返回结果
            unique_results = list(dict.fromkeys(result_texts))  # 保持顺序去重

            if not unique_results:
                return "未抓取到有效结果，可能是 Google 返回了验证码或页面结构变化"

            # 返回前 20 条结果
            return "\n".join(unique_results[:20])

    except Exception as e:
        if browser:
            try:
                browser.close()
            except:
                pass
        return f"Google Lens 搜索失败: {str(e)}"


def extract_brand_and_price(raw_text: str) -> Dict[str, List[str]]:
    """
    从原始文本中提取品牌和价格信息

    Args:
        raw_text: Google Lens 抓取的原始文本

    Returns:
        {"brands": [...], "prices": [...]}
    """
    brands = []
    prices = []

    # 价格匹配（多种格式）
    price_patterns = [
        r'¥\s*[\d,]+(?:\.\d+)?',  # ¥123, ¥1,234.50
        r'\$\s*[\d,]+(?:\.\d+)?',  # $123
        r'[\d,]+\s*元',  # 123元
        r'[\d,]+(?:\.\d+)?\s*USD',
        r'price[:\s]*[\d,]+',
    ]

    for pattern in price_patterns:
        matches = re.findall(pattern, raw_text, re.IGNORECASE)
        prices.extend(matches)

    # 品牌匹配（大写字母开头的单词，或者常见品牌词）
    # 这里用简单启发式：长度适中、首字母大写的词可能是品牌
    words = re.findall(r'\b[A-Z][a-zA-Z]{2,10}\b', raw_text)
    brand_blacklist = ['The', 'And', 'For', 'With', 'Your', 'This', 'That', 'From', 'Search', 'Images', 'Shopping']
    for word in words:
        if word not in brand_blacklist and word not in brands:
            brands.append(word)

    return {
        "brands": brands[:5],  # 最多5个
        "prices": list(set(prices))[:5]  # 去重后最多5个
    }


# 测试入口
if __name__ == "__main__":
    # 创建一个测试用的假图片
    from PIL import Image

    test_image_path = "/tmp/test_clothing.jpg"

    # 生成一张测试图（蓝白条纹，模拟衣服）
    img = Image.new('RGB', (400, 600), color='white')
    pixels = img.load()
    for y in range(600):
        for x in range(400):
            if (y // 20) % 2 == 0:
                pixels[x, y] = (70, 130, 180)  # 钢蓝色条纹
    img.save(test_image_path)
    print(f"[Test] 已创建测试图片: {test_image_path}")

    # 执行搜索
    print("\n" + "="*50)
    print("开始 Google Lens 搜索测试...")
    print("="*50 + "\n")

    result = search_clothing_on_google_lens(test_image_path, headless=False)

    print("\n" + "="*50)
    print("搜索结果:")
    print("="*50)
    print(result)

    # 提取品牌和价格
    print("\n" + "="*50)
    print("提取信息:")
    print("="*50)
    extracted = extract_brand_and_price(result)
    print(f"可能的品牌: {extracted['brands']}")
    print(f"可能的价格: {extracted['prices']}")
