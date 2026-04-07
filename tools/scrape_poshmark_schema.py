"""
简化版 Poshmark 分类和尺码抓取
只抓取常用分类：
- Women > Tops, Dresses, Jackets, Pants
- Men > Tops, Pants
"""
import json
import time
from playwright.sync_api import sync_playwright

USER_DATA_DIR = "./poshmark_browser_data"

# 只抓这些常用分类
TARGET_CATEGORIES = {
    "Women": ["Tops", "Dresses", "Jackets & Coats", "Pants"],
    "Men": ["Tops", "Pants"]
}


def scrape_poshmark_schema():
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            USER_DATA_DIR,
            headless=False,  # 必须 False，需要手动登录
            args=['--disable-blink-features=AutomationControlled']
        )
        page = browser.new_page()
        page.goto("https://poshmark.com/create-listing")
        page.wait_for_load_state("networkidle")

        # 检查是否需要登录
        if "login" in page.url:
            print("🔐 请在浏览器中手动登录 Poshmark...")
            print("登录完成后，脚本会自动继续")
            # 等待登录完成
            for i in range(120):  # 最多等2分钟
                if "create-listing" in page.url:
                    print("✅ 登录成功！")
                    break
                time.sleep(1)
            else:
                print("❌ 登录超时")
                browser.close()
                return

        schema = {}

        for primary_cat, secondary_cats in TARGET_CATEGORIES.items():
            schema[primary_cat] = {}
            print(f"\n📂 正在抓取大类: {primary_cat}")

            for secondary_cat in secondary_cats:
                print(f"  └─ 子类: {secondary_cat}")

                try:
                    # 刷新页面重新开始（避免状态混乱）
                    page.goto("https://poshmark.com/create-listing")
                    page.wait_for_load_state("networkidle")
                    time.sleep(2)

                    # 1. 触发 Category 选择器
                    page.get_by_text("Select Category").first.click()
                    time.sleep(1.5)

                    # 2. 点击一级分类
                    page.get_by_text(primary_cat, exact=True).filter(visible=True).first.click()
                    time.sleep(1)

                    # 3. 点击二级分类（支持模糊匹配）
                    # 先尝试精确匹配，失败则尝试包含匹配
                    clicked = False
                    try:
                        page.get_by_text(secondary_cat, exact=True).filter(visible=True).first.click()
                        clicked = True
                    except:
                        # 尝试模糊匹配（如 "Pants" 匹配 "Pants & Jumpsuits"）
                        try:
                            page.get_by_text(secondary_cat, exact=False).filter(visible=True).first.click()
                            clicked = True
                        except:
                            # 最后尝试滚动后点击
                            page.evaluate(f"""
                                const items = document.querySelectorAll('*');
                                for (const item of items) {{
                                    if (item.textContent && item.textContent.includes('{secondary_cat}')) {{
                                        item.scrollIntoView({{behavior: 'instant', block: 'center'}});
                                        item.click();
                                        break;
                                    }}
                                }}
                            """)
                            clicked = True

                    if not clicked:
                        raise Exception(f"无法点击二级分类: {secondary_cat}")
                    time.sleep(0.8)

                    # 4. 关闭分类选择器
                    page.keyboard.press("Escape")
                    time.sleep(0.5)

                    # 5. 抓取该分类对应的尺码 (Size)
                    print(f"      正在抓取尺码...")
                    page.get_by_text("Select Size").first.click()
                    time.sleep(1.5)

                    # 尝试多种可能的尺码选择器
                    sizes = []
                    for selector in [
                        '.popover-content .grid__item',
                        '.modal-content button',
                        '[data-testid="size-grid"] button',
                        '.size-selector button'
                    ]:
                        try:
                            elements = page.locator(selector).all_inner_texts()
                            if elements:
                                sizes = [s.strip() for s in elements if s.strip()]
                                break
                        except:
                            continue

                    schema[primary_cat][secondary_cat] = {
                        "sizes": sizes if sizes else ["OS"]  # 如果没抓到，默认OS
                    }
                    print(f"      ✅ 尺码: {sizes if sizes else ['OS']}")

                    # 关闭 Size 弹窗
                    page.keyboard.press("Escape")
                    time.sleep(0.3)

                except Exception as e:
                    print(f"      ⚠️ 抓取失败: {e}")
                    schema[primary_cat][secondary_cat] = {"sizes": ["OS"], "error": str(e)}

        # 保存结果
        output_file = 'poshmark_schema_lite.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 抓取完成！已保存至 {output_file}")
        print(json.dumps(schema, ensure_ascii=False, indent=2))
        browser.close()


if __name__ == "__main__":
    scrape_poshmark_schema()
